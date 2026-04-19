"""
BigVGAN fine-tuning script for kNN-VC.

Fine-tunes a pretrained BigVGAN v2 (24kHz, 100-band mel) to accept WavLM features
instead of mel spectrograms. A learned projection layer maps the 1024-dim WavLM
features to the 100-dim pseudo-mel space expected by BigVGAN.

Training Phases:
  Phase A (~50k steps): Train projection layer only, BigVGAN backbone frozen.
  Phase B (~100k-300k steps): Unfreeze BigVGAN, train end-to-end at lower LR.

Supports single-GPU and multi-GPU (single-node) training via torchrun.

Usage:
  # Extract WavLM features first:
  python scripts/extract_wavlm_features.py --audio_dir /path/to/audio --out_dir /path/to/feats

  # Phase A: Train projection only (single GPU)
  python train_bigvgan.py \\
      --audio_dir /path/to/audio \\
      --feat_dir /path/to/feats \\
      --checkpoint_dir ./checkpoints/bigvgan \\
      --phase A \\
      --steps 50000

  # Phase A: Train projection only (multi-GPU via torchrun)
  torchrun --nproc_per_node=4 train_bigvgan.py \\
      --audio_dir /path/to/audio \\
      --feat_dir /path/to/feats \\
      --checkpoint_dir ./checkpoints/bigvgan \\
      --phase A \\
      --steps 50000

  # Phase B: Fine-tune end-to-end (start from Phase A checkpoint)
  torchrun --nproc_per_node=4 train_bigvgan.py \\
      --audio_dir /path/to/audio \\
      --feat_dir /path/to/feats \\
      --checkpoint_dir ./checkpoints/bigvgan \\
      --phase B \\
      --resume ./checkpoints/bigvgan/ckpt_050000.pt \\
      --steps 300000

Dataset format:
  Audio: 24kHz .wav files (or any SR, will be resampled)
  Features: Corresponding .pt files with WavLM layer-6 features, shape (seq_len, 1024)
  File pairing: audio at feat_dir/<relpath>.pt for audio at audio_dir/<relpath>.wav
"""

import argparse
import itertools
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import bigvgan as bigvgan_module
from bigvgan.discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiScaleSubbandCQTDiscriminator,
    MultiBandDiscriminator,
)
from bigvgan.loss import (
    MultiScaleMelSpectrogramLoss,
    discriminator_loss,
    feature_loss,
    generator_loss,
)
from bigvgan_vocoder import BigVGANVocoder


# ──────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ──────────────────────────────────────────────────────────────────────────────

def setup_distributed():
    """Initialize distributed training if launched via torchrun. Returns (rank, local_rank, world_size)."""
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class WavLMVocoderDataset(Dataset):
    """Paired (WavLM features, audio waveform) dataset for BigVGAN fine-tuning.

    Expects:
      - audio_dir: directory tree of .wav files at target_sr
      - feat_dir: directory tree of .pt files (WavLM layer-6 features, shape (seq_len, 1024))
        with the same relative paths as audio files but .pt extension

    Segment size is in samples at target_sr. Features are trimmed/padded to match.
    """

    def __init__(
        self,
        audio_dir: Path,
        feat_dir: Path,
        segment_size: int = 24000 * 1,  # 1 second at 24kHz
        target_sr: int = 24000,
        wavlm_frame_rate: int = 50,  # 50 frames/sec = 20ms hop
        split: bool = True,
    ):
        self.audio_dir = Path(audio_dir)
        self.feat_dir = Path(feat_dir)
        self.segment_size = segment_size
        self.target_sr = target_sr
        self.wavlm_frame_rate = wavlm_frame_rate
        self.split = split
        self.frames_per_seg = math.ceil(segment_size / target_sr * wavlm_frame_rate)

        # Find all paired files
        self.pairs = []
        for feat_path in sorted(self.feat_dir.rglob('*.pt')):
            rel = feat_path.relative_to(self.feat_dir).with_suffix('.wav')
            audio_path = self.audio_dir / rel
            if audio_path.exists():
                self.pairs.append((audio_path, feat_path))

        if len(self.pairs) == 0:
            raise ValueError(
                f"No paired (audio, feature) files found.\n"
                f"  audio_dir: {audio_dir}\n"
                f"  feat_dir: {feat_dir}\n"
                "Run scripts/extract_wavlm_features.py first."
            )
        print(f"[Dataset] Found {len(self.pairs):,d} paired files.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio_path, feat_path = self.pairs[idx]

        # Load audio
        wav, sr = torchaudio.load(audio_path, normalize=True)
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        wav = wav.mean(dim=0)  # mono, shape (T,)

        # Load WavLM features
        feats = torch.load(feat_path, map_location='cpu').float()  # (seq_len, 1024)

        if self.split:
            # Pad short utterances so every sample is exactly (frames_per_seg, 1024) / (segment_size,)
            if feats.shape[0] < self.frames_per_seg:
                feats = F.pad(feats, (0, 0, 0, self.frames_per_seg - feats.shape[0]))
            if wav.shape[0] < self.segment_size:
                wav = F.pad(wav, (0, self.segment_size - wav.shape[0]))

            # Random crop aligned between features and audio
            max_feat_start = max(0, feats.shape[0] - self.frames_per_seg)
            feat_start = random.randint(0, max_feat_start)
            feat_end = feat_start + self.frames_per_seg

            # Convert feat frame to audio sample (approximate, WavLM at 50Hz)
            audio_start = int(feat_start / self.wavlm_frame_rate * self.target_sr)
            audio_end = audio_start + self.segment_size

            feats = feats[feat_start:feat_end]  # (frames_per_seg, 1024)

            if audio_end <= wav.shape[0]:
                wav = wav[audio_start:audio_end]
            else:
                wav = wav[audio_start:]
                wav = F.pad(wav, (0, self.segment_size - wav.shape[0]))
        else:
            # Validation: pad feats to frames_per_seg
            if feats.shape[0] < self.frames_per_seg:
                feats = F.pad(feats, (0, 0, 0, self.frames_per_seg - feats.shape[0]))
            if wav.shape[0] < self.segment_size:
                wav = F.pad(wav, (0, self.segment_size - wav.shape[0]))

        return feats, wav  # (frames_per_seg, 1024), (segment_size,)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, vocoder, mpd, mrd, optim_g, optim_d, steps, epoch):
    # Unwrap DDP modules to save the underlying state_dict
    def unwrap(m):
        return m.module if isinstance(m, DDP) else m

    torch.save({
        'projection': unwrap(vocoder).projection.state_dict(),
        'bigvgan': unwrap(vocoder).bigvgan.state_dict(),
        'mpd': unwrap(mpd).state_dict(),
        'mrd': unwrap(mrd).state_dict(),
        'optim_g': optim_g.state_dict(),
        'optim_d': optim_d.state_dict(),
        'steps': steps,
        'epoch': epoch,
    }, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main_process(rank):
        print(f"[Train] world_size={world_size}, device={device}")

    # ── Load pretrained BigVGAN ──────────────────────────────────────────────
    bigvgan_model = bigvgan_module.BigVGAN._from_pretrained(
        model_id='nvidia/bigvgan_v2_24khz_100band_256x',
        revision=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=False,
        token=None,
        use_cuda_kernel=False,
        map_location=str(device),
    )
    h = bigvgan_model.h  # hyperparams: sampling_rate=24000, num_mels=100, hop_size=256, ...

    projection = nn.Linear(1024, h.num_mels)
    vocoder = BigVGANVocoder(bigvgan_model, projection, target_sr=h.sampling_rate).to(device)

    # ── Discriminators (match BigVGAN-v2 config) ─────────────────────────────
    mpd = MultiPeriodDiscriminator(h).to(device)
    if h.get('use_mbd_instead_of_mrd', False):
        mrd = MultiBandDiscriminator(h).to(device)
    elif h.get('use_cqtd_instead_of_mrd', False):
        mrd = MultiScaleSubbandCQTDiscriminator(h).to(device)
    else:
        mrd = MultiResolutionDiscriminator(h).to(device)

    # ── Resume from checkpoint ───────────────────────────────────────────────
    steps = 0
    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        vocoder.projection.load_state_dict(ckpt['projection'])
        vocoder.bigvgan.load_state_dict(ckpt['bigvgan'])
        mpd.load_state_dict(ckpt['mpd'])
        mrd.load_state_dict(ckpt['mrd'])
        steps = ckpt.get('steps', 0)
        start_epoch = ckpt.get('epoch', 0)
        if is_main_process(rank):
            print(f"[Train] Resumed from {args.resume} at step {steps}")

    # ── Freeze/unfreeze based on phase ───────────────────────────────────────
    if args.phase == 'A':
        if is_main_process(rank):
            print("[Train] Phase A: training projection layer only, BigVGAN frozen.")
        for p in vocoder.bigvgan.parameters():
            p.requires_grad = False
        g_params = list(vocoder.projection.parameters())
        lr_g = 2e-4
    else:
        if is_main_process(rank):
            print("[Train] Phase B: end-to-end fine-tuning.")
        for p in vocoder.bigvgan.parameters():
            p.requires_grad = True
        g_params = list(vocoder.parameters())
        lr_g = 1e-4  # Lower LR for end-to-end fine-tuning

    # ── Wrap models with DDP ─────────────────────────────────────────────────
    if world_size > 1:
        vocoder = DDP(vocoder, device_ids=[local_rank], find_unused_parameters=True)
        mpd = DDP(mpd, device_ids=[local_rank])
        mrd = DDP(mrd, device_ids=[local_rank])

    # ── Optimizers ───────────────────────────────────────────────────────────
    optim_g = torch.optim.AdamW(g_params, lr=lr_g, betas=(0.8, 0.99))
    optim_d = torch.optim.AdamW(
        itertools.chain(mpd.parameters(), mrd.parameters()),
        lr=lr_g, betas=(0.8, 0.99)
    )

    if args.resume:
        optim_g.load_state_dict(ckpt['optim_g'])
        optim_d.load_state_dict(ckpt['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.999, last_epoch=start_epoch - 1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.999, last_epoch=start_epoch - 1)

    # ── Loss functions ────────────────────────────────────────────────────────
    mel_loss_fn = MultiScaleMelSpectrogramLoss(sampling_rate=h.sampling_rate).to(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    segment_size = h.sampling_rate * args.segment_seconds  # samples at 24kHz
    dataset = WavLMVocoderDataset(
        audio_dir=args.audio_dir,
        feat_dir=args.feat_dir,
        segment_size=segment_size,
        target_sr=h.sampling_rate,
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == 'cuda'),
    )

    # ── Tensorboard (main process only) ───────────────────────────────────────
    sw = None
    if is_main_process(rank):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        sw = SummaryWriter(os.path.join(args.checkpoint_dir, 'logs'))

    # ── Training loop ─────────────────────────────────────────────────────────
    vocoder.train()
    mpd.train()
    mrd.train()

    for epoch in range(start_epoch, 10000):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for feats, wav_real in loader:
            feats = feats.to(device)       # (B, frames, 1024)
            wav_real = wav_real.to(device).unsqueeze(1)  # (B, 1, T)

            # Forward pass: WavLM features → synthesized audio (at 24kHz, no resample)
            wav_gen = vocoder(feats)  # (B, 1, T')

            # Align lengths (resampling may cause off-by-one)
            min_len = min(wav_real.shape[-1], wav_gen.shape[-1])
            wav_real = wav_real[..., :min_len]
            wav_gen = wav_gen[..., :min_len]

            # ── Discriminator update ────────────────────────────────────────
            optim_d.zero_grad()

            y_df_r, y_df_g, _, _ = mpd(wav_real, wav_gen.detach())
            loss_d_mpd, _, _ = discriminator_loss(y_df_r, y_df_g)

            y_dr_r, y_dr_g, _, _ = mrd(wav_real, wav_gen.detach())
            loss_d_mrd, _, _ = discriminator_loss(y_dr_r, y_dr_g)

            loss_d = loss_d_mpd + loss_d_mrd
            loss_d.backward()
            optim_d.step()

            # ── Generator update ────────────────────────────────────────────
            optim_g.zero_grad()

            # Mel-spectrogram loss (multi-scale)
            loss_mel = mel_loss_fn(wav_gen, wav_real)

            # Adversarial + feature matching losses
            y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(wav_real, wav_gen)
            y_dr_r, y_dr_g, fmap_r_r, fmap_r_g = mrd(wav_real, wav_gen)

            loss_fm_mpd = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_mrd = feature_loss(fmap_r_r, fmap_r_g)
            loss_gen_mpd, _ = generator_loss(y_df_g)
            loss_gen_mrd, _ = generator_loss(y_dr_g)

            loss_g = loss_mel + loss_fm_mpd + loss_fm_mrd + loss_gen_mpd + loss_gen_mrd
            loss_g.backward()
            optim_g.step()

            # ── Logging (main process only) ─────────────────────────────────
            if is_main_process(rank) and steps % args.log_interval == 0:
                print(
                    f"Step {steps:,d} | "
                    f"G={loss_g.item():.3f} mel={loss_mel.item():.3f} "
                    f"fm={loss_fm_mpd.item()+loss_fm_mrd.item():.3f} "
                    f"D={loss_d.item():.3f}"
                )
                sw.add_scalar('train/loss_g', loss_g.item(), steps)
                sw.add_scalar('train/loss_mel', loss_mel.item(), steps)
                sw.add_scalar('train/loss_d', loss_d.item(), steps)

            # ── Checkpoint (main process only) ──────────────────────────────
            if is_main_process(rank) and steps % args.save_interval == 0 and steps > 0:
                ckpt_path = Path(args.checkpoint_dir) / f'ckpt_{steps:06d}.pt'
                save_checkpoint(ckpt_path, vocoder, mpd, mrd, optim_g, optim_d, steps, epoch)
                print(f"[Train] Saved checkpoint: {ckpt_path}")

            steps += 1
            if steps >= args.steps:
                if is_main_process(rank):
                    print(f"[Train] Reached {args.steps:,d} steps. Done.")
                    ckpt_path = Path(args.checkpoint_dir) / f'ckpt_{steps:06d}_final.pt'
                    save_checkpoint(ckpt_path, vocoder, mpd, mrd, optim_g, optim_d, steps, epoch)
                cleanup_distributed()
                return

        scheduler_g.step()
        scheduler_d.step()

    cleanup_distributed()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Fine-tune BigVGAN for WavLM-VC')
    parser.add_argument('--audio_dir', required=True, help='Root directory of .wav files at 24kHz')
    parser.add_argument('--feat_dir', required=True, help='Root directory of .pt WavLM feature files')
    parser.add_argument('--checkpoint_dir', default='./checkpoints/bigvgan')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--phase', choices=['A', 'B'], default='A',
                        help='A=projection only, B=end-to-end fine-tune')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--segment_seconds', type=int, default=1,
                        help='Audio segment length in seconds for training crops')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5000)
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
