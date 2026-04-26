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
import contextlib
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
from bigvgan_vocoder import BigVGANVocoder, build_projection


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


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class WavLMVocoderDataset(Dataset):
    """Paired (WavLM features, audio waveform) dataset for BigVGAN fine-tuning.

    Expects one or more (audio_dir, feat_dir) pairs. Each feat_dir contains .pt files
    (WavLM layer-6 features, shape (seq_len, 1024)) with the same relative paths as
    the corresponding audio files but with .pt extension.

    Segment size is in samples at target_sr. Features are trimmed/padded to match.

    When supplementary directories are provided, `group_indices` tracks which samples
    belong to the primary (0) vs supplementary (1) group for weighted sampling.
    """

    AUDIO_EXTS = ('.wav', '.flac', '.mp3', '.ogg', '.opus')

    def __init__(
        self,
        audio_dir: Path,
        feat_dir: Path,
        supplementary_dirs: list[tuple[Path, Path]] | None = None,
        segment_size: int = 24000 * 1,  # 1 second at 24kHz
        target_sr: int = 24000,
        wavlm_frame_rate: int = 50,  # 50 frames/sec = 20ms hop
        split: bool = True,
    ):
        self.segment_size = segment_size
        self.target_sr = target_sr
        self.wavlm_frame_rate = wavlm_frame_rate
        self.split = split
        self.frames_per_seg = math.ceil(segment_size / target_sr * wavlm_frame_rate)

        self.pairs = []
        self.group_of = []  # 0 = primary, 1 = supplementary

        self._scan_dirs(Path(audio_dir), Path(feat_dir), group=0)
        for sup_audio, sup_feat in (supplementary_dirs or []):
            self._scan_dirs(Path(sup_audio), Path(sup_feat), group=1)

        if len(self.pairs) == 0:
            raise ValueError(
                f"No paired (audio, feature) files found.\n"
                f"  audio_dir: {audio_dir}\n"
                f"  feat_dir: {feat_dir}\n"
                "Run scripts/extract_wavlm_features.py first."
            )

        n_primary = self.group_of.count(0)
        n_supp = self.group_of.count(1)
        log(f"[Dataset] {n_primary:,d} primary + {n_supp:,d} supplementary = {len(self.pairs):,d} total pairs.")

    def _scan_dirs(self, audio_dir: Path, feat_dir: Path, group: int):
        missing_audio = 0
        found = 0
        for feat_path in sorted(feat_dir.rglob('*.pt')):
            rel_no_ext = feat_path.relative_to(feat_dir).with_suffix('')
            audio_path = None
            for ext in self.AUDIO_EXTS:
                candidate = audio_dir / rel_no_ext.with_suffix(ext)
                if candidate.exists():
                    audio_path = candidate
                    break
            if audio_path is not None:
                self.pairs.append((audio_path, feat_path))
                self.group_of.append(group)
                found += 1
            else:
                missing_audio += 1
        if missing_audio:
            log(f"[Dataset] WARNING: {missing_audio:,d} .pt files have no matching audio in {audio_dir}.")
        label = "primary" if group == 0 else "supplementary"
        log(f"[Dataset] Found {found:,d} paired files in {feat_dir} ({label}).")

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
# Weighted sampler for supplementary data
# ──────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(dataset, supplementary_weight, rank=0, world_size=1, epoch=0):
    """Build a WeightedRandomSampler that oversamples supplementary data.

    Args:
        dataset: WavLMVocoderDataset with group_of attribute.
        supplementary_weight: Target fraction of supplementary data per epoch (0.0-1.0).
            E.g. 0.15 means ~15% of samples per epoch come from supplementary data.
        rank, world_size: For distributed training, each rank gets a disjoint subset.
        epoch: Used as random seed offset for reproducibility across epochs.
    """
    n_primary = dataset.group_of.count(0)
    n_supp = dataset.group_of.count(1)

    if n_supp == 0 or supplementary_weight <= 0:
        return None

    # Per-sample weights so that P(supplementary) = supplementary_weight in expectation.
    # With n_prim samples at weight w_prim and n_supp at w_supp:
    #   P(supp) = n_supp * w_supp / (n_prim * w_prim + n_supp * w_supp) = supplementary_weight
    w_prim = 1.0
    w_supp = (supplementary_weight * n_primary) / ((1 - supplementary_weight) * n_supp)

    weights = [w_supp if g == 1 else w_prim for g in dataset.group_of]
    weights = torch.tensor(weights, dtype=torch.double)

    n_samples = len(dataset)

    if world_size > 1:
        # Each rank draws n_samples // world_size samples with the same weights
        # but different random seeds, giving disjoint-in-expectation subsets.
        n_samples = math.ceil(len(dataset) / world_size)
        g = torch.Generator()
        g.manual_seed(epoch * world_size + rank)
        return torch.utils.data.WeightedRandomSampler(weights, num_samples=n_samples, replacement=True, generator=g)

    return torch.utils.data.WeightedRandomSampler(weights, num_samples=n_samples, replacement=True)


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, vocoder, mpd, mrd, optim_g, optim_d, steps, epoch, phase,
                    projection_type='linear'):
    # Unwrap DDP modules to save the underlying state_dict
    def unwrap(m):
        return m.module if isinstance(m, DDP) else m

    # Atomic write: save to .tmp then rename, so an interrupted save doesn't leave
    # a half-written checkpoint on disk.
    tmp_path = str(path) + '.tmp'
    torch.save({
        'projection': unwrap(vocoder).projection.state_dict(),
        'projection_type': projection_type,
        'bigvgan': unwrap(vocoder).bigvgan.state_dict(),
        'mpd': unwrap(mpd).state_dict(),
        'mrd': unwrap(mrd).state_dict(),
        'optim_g': optim_g.state_dict(),
        'optim_d': optim_d.state_dict(),
        'steps': steps,
        'epoch': epoch,
        'phase': phase,
    }, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def prune_old_checkpoints(checkpoint_dir: Path, keep_last: int):
    """Keep only the `keep_last` most-recent periodic checkpoints (ckpt_NNNNNN.pt).
    Never touches *_final.pt or files with other naming."""
    if keep_last <= 0:
        return
    ckpts = sorted(
        checkpoint_dir.glob('ckpt_[0-9]*.pt'),
        key=lambda p: p.stat().st_mtime,
    )
    # Keep only periodic (not final) checkpoints for pruning purposes
    periodic = [p for p in ckpts if not p.name.endswith('_final.pt')]
    for old in periodic[:-keep_last]:
        try:
            old.unlink()
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(vocoder, mel_loss_fn, val_loader, device, sw, steps, num_audio_samples=4):
    """Run validation: compute mel loss over the val set, log to TensorBoard."""
    vocoder.eval()

    total_mel = 0.0
    n_batches = 0
    audio_logged = 0

    for feats, wav_real in val_loader:
        feats = feats.to(device)
        wav_real = wav_real.to(device).unsqueeze(1)

        wav_gen = vocoder(feats)

        min_len = min(wav_real.shape[-1], wav_gen.shape[-1])
        wav_real = wav_real[..., :min_len]
        wav_gen = wav_gen[..., :min_len]

        total_mel += mel_loss_fn(wav_gen, wav_real).item()
        n_batches += 1

        if sw is not None and audio_logged < num_audio_samples:
            for j in range(min(feats.shape[0], num_audio_samples - audio_logged)):
                sw.add_audio(f'val/gen_{audio_logged}', wav_gen[j], steps, sample_rate=24000)
                sw.add_audio(f'val/real_{audio_logged}', wav_real[j], steps, sample_rate=24000)
                audio_logged += 1

    val_mel = total_mel / max(n_batches, 1)

    if sw is not None:
        sw.add_scalar('val/loss_mel', val_mel, steps)

    vocoder.train()
    return val_mel


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main_process(rank):
        log(f"[Train] world_size={world_size}, device={device}")

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

    projection = build_projection(args.projection, in_dim=1024, out_dim=h.num_mels)
    vocoder = BigVGANVocoder(bigvgan_model, projection, target_sr=h.sampling_rate).to(device)
    if is_main_process(rank):
        n_proj = sum(p.numel() for p in projection.parameters())
        log(f"[Train] Projection: {args.projection} ({n_proj:,d} params)")

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
    ckpt = None
    resumed_phase = None
    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        ckpt_proj_type = ckpt.get('projection_type', 'linear')
        if ckpt_proj_type != args.projection:
            raise ValueError(
                f"Checkpoint uses projection '{ckpt_proj_type}' but --projection={args.projection}. "
                f"Use --projection {ckpt_proj_type} to match the checkpoint.")
        vocoder.projection.load_state_dict(ckpt['projection'])
        vocoder.bigvgan.load_state_dict(ckpt['bigvgan'])
        mpd.load_state_dict(ckpt['mpd'])
        mrd.load_state_dict(ckpt['mrd'])
        steps = ckpt.get('steps', 0)
        start_epoch = ckpt.get('epoch', 0)
        resumed_phase = ckpt.get('phase', None)
        if is_main_process(rank):
            log(f"[Train] Resumed from {args.resume} at step {steps} (phase={resumed_phase})")

    # ── Freeze/unfreeze based on phase ───────────────────────────────────────
    if args.phase == 'A':
        if is_main_process(rank):
            log("[Train] Phase A: training projection layer only, BigVGAN frozen.")
        for p in vocoder.bigvgan.parameters():
            p.requires_grad = False
        g_params = list(vocoder.projection.parameters())
        default_lr = 2e-4
    else:
        if is_main_process(rank):
            log("[Train] Phase B: end-to-end fine-tuning.")
        for p in vocoder.bigvgan.parameters():
            p.requires_grad = True
        g_params = list(vocoder.parameters())
        default_lr = 1e-4

    lr_g = args.lr_g if args.lr_g is not None else default_lr
    lr_d = args.lr_d if args.lr_d is not None else lr_g
    if is_main_process(rank):
        log(f"[Train] LR: generator={lr_g:.1e}, discriminator={lr_d:.1e}")

    # ── Wrap models with DDP ─────────────────────────────────────────────────
    if world_size > 1:
        vocoder = DDP(vocoder, device_ids=[local_rank], find_unused_parameters=True)
        mpd = DDP(mpd, device_ids=[local_rank])
        mrd = DDP(mrd, device_ids=[local_rank])

    # ── Optimizers ───────────────────────────────────────────────────────────
    optim_g = torch.optim.AdamW(g_params, lr=lr_g, betas=(0.8, 0.99))
    optim_d = torch.optim.AdamW(
        itertools.chain(mpd.parameters(), mrd.parameters()),
        lr=lr_d, betas=(0.8, 0.99)
    )

    if args.resume:
        # Only restore optimizer state when continuing the SAME phase. A Phase A→B
        # transition changes the generator's parameter set, so the saved optim_g
        # state is incompatible.
        same_phase = (resumed_phase == args.phase)
        if same_phase:
            optim_g.load_state_dict(ckpt['optim_g'])
            optim_d.load_state_dict(ckpt['optim_d'])
        else:
            if is_main_process(rank):
                log(f"[Train] Phase change (ckpt={resumed_phase} → run={args.phase}): "
                    f"re-initializing optimizer states with fresh LR={lr_g:.1e}.")

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.999, last_epoch=max(-1, start_epoch - 1))
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.999, last_epoch=max(-1, start_epoch - 1))

    # ── Loss functions ────────────────────────────────────────────────────────
    mel_loss_fn = MultiScaleMelSpectrogramLoss(sampling_rate=h.sampling_rate).to(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    segment_size = h.sampling_rate * args.segment_seconds  # samples at 24kHz

    supplementary_dirs = None
    if args.supplementary_dirs:
        supplementary_dirs = []
        for pair in args.supplementary_dirs:
            a_dir, f_dir = pair.split(':')
            supplementary_dirs.append((Path(a_dir), Path(f_dir)))

    dataset = WavLMVocoderDataset(
        audio_dir=args.audio_dir,
        feat_dir=args.feat_dir,
        supplementary_dirs=supplementary_dirs,
        segment_size=segment_size,
        target_sr=h.sampling_rate,
    )

    has_supplementary = dataset.group_of.count(1) > 0
    if has_supplementary:
        sampler = make_weighted_sampler(dataset, args.supplementary_weight, rank, world_size)
        if is_main_process(rank):
            log(f"[Train] Weighted sampling: supplementary target weight = {args.supplementary_weight:.0%}")
    elif world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(device.type == 'cuda'),
    )

    # ── Validation dataset ──────────────────────────────────────────────────
    val_loader = None
    if args.val_audio_dir and args.val_feat_dir:
        val_dataset = WavLMVocoderDataset(
            audio_dir=args.val_audio_dir,
            feat_dir=args.val_feat_dir,
            segment_size=segment_size,
            target_sr=h.sampling_rate,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=(device.type == 'cuda'),
        )
        if is_main_process(rank):
            log(f"[Train] Validation enabled: {len(val_dataset):,d} samples, "
                f"running every {args.val_interval:,d} steps.")

    # ── Tensorboard (main process only) ───────────────────────────────────────
    sw = None
    if is_main_process(rank):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        sw = SummaryWriter(os.path.join(args.checkpoint_dir, 'logs'))

    # ── Training loop ─────────────────────────────────────────────────────────
    if is_main_process(rank):
        eff_bs = args.batch_size * world_size * args.accumulation_steps
        log(f"[Train] Effective batch size: {args.batch_size} x {world_size} GPUs x "
            f"{args.accumulation_steps} accum = {eff_bs}")

    vocoder.train()
    mpd.train()
    mrd.train()

    accum_steps = args.accumulation_steps
    use_ddp = world_size > 1

    # no_sync() skips DDP all-reduce on intermediate micro-batches
    def maybe_no_sync(model, is_last):
        if use_ddp and not is_last:
            return model.no_sync()
        return contextlib.nullcontext()

    optim_d.zero_grad()
    optim_g.zero_grad()

    for epoch in range(start_epoch, 10000):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        elif has_supplementary:
            sampler = make_weighted_sampler(dataset, args.supplementary_weight, rank, world_size, epoch)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                drop_last=True,
                pin_memory=(device.type == 'cuda'),
            )

        micro = 0
        accum_loss_g = 0.0
        accum_loss_d = 0.0
        accum_loss_mel = 0.0
        accum_loss_fm = 0.0

        for feats, wav_real in loader:
            feats = feats.to(device)       # (B, frames, 1024)
            wav_real = wav_real.to(device).unsqueeze(1)  # (B, 1, T)
            micro += 1
            is_last_micro = (micro % accum_steps == 0)

            # Forward pass: WavLM features → synthesized audio (at 24kHz, no resample)
            wav_gen = vocoder(feats)  # (B, 1, T')

            # Align lengths (resampling may cause off-by-one)
            min_len = min(wav_real.shape[-1], wav_gen.shape[-1])
            wav_real = wav_real[..., :min_len]
            wav_gen = wav_gen[..., :min_len]

            # ── Discriminator update ────────────────────────────────────────
            with maybe_no_sync(mpd, is_last_micro), maybe_no_sync(mrd, is_last_micro):
                y_df_r, y_df_g, _, _ = mpd(wav_real, wav_gen.detach())
                loss_d_mpd, _, _ = discriminator_loss(y_df_r, y_df_g)

                y_dr_r, y_dr_g, _, _ = mrd(wav_real, wav_gen.detach())
                loss_d_mrd, _, _ = discriminator_loss(y_dr_r, y_dr_g)

                loss_d = (loss_d_mpd + loss_d_mrd) / accum_steps
                loss_d.backward()

            # ── Generator update ────────────────────────────────────────────
            with maybe_no_sync(vocoder, is_last_micro), \
                 maybe_no_sync(mpd, is_last_micro), maybe_no_sync(mrd, is_last_micro):
                loss_mel = mel_loss_fn(wav_gen, wav_real)

                y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(wav_real, wav_gen)
                y_dr_r, y_dr_g, fmap_r_r, fmap_r_g = mrd(wav_real, wav_gen)

                loss_fm_mpd = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_mrd = feature_loss(fmap_r_r, fmap_r_g)
                loss_gen_mpd, _ = generator_loss(y_df_g)
                loss_gen_mrd, _ = generator_loss(y_dr_g)

                loss_g = (loss_mel + loss_fm_mpd + loss_fm_mrd + loss_gen_mpd + loss_gen_mrd) / accum_steps
                loss_g.backward()

            # Track unscaled losses for logging
            accum_loss_d += loss_d.item() * accum_steps
            accum_loss_g += loss_g.item() * accum_steps
            accum_loss_mel += loss_mel.item()
            accum_loss_fm += (loss_fm_mpd.item() + loss_fm_mrd.item())

            # Stop training immediately if we hit a non-finite loss — for long
            # runs this saves hours of wasted compute chasing NaNs.
            if not (torch.isfinite(loss_g) and torch.isfinite(loss_d)):
                if is_main_process(rank):
                    log(f"[Train] ABORT: non-finite loss at step {steps} "
                        f"(G={loss_g.item()}, D={loss_d.item()})")
                cleanup_distributed()
                return

            if not is_last_micro:
                continue

            # ── Optimizer step (every accum_steps micro-batches) ───────────
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    itertools.chain(mpd.parameters(), mrd.parameters()),
                    args.grad_clip,
                )
                torch.nn.utils.clip_grad_norm_(g_params, args.grad_clip)
            optim_d.step()
            optim_g.step()
            optim_d.zero_grad()
            optim_g.zero_grad()

            # ── Logging (main process only) ─────────────────────────────────
            if is_main_process(rank) and steps % args.log_interval == 0:
                avg_mel = accum_loss_mel / accum_steps
                avg_fm = accum_loss_fm / accum_steps
                avg_g = accum_loss_g / accum_steps
                avg_d = accum_loss_d / accum_steps
                log(
                    f"Step {steps:,d} | "
                    f"G={avg_g:.3f} mel={avg_mel:.3f} "
                    f"fm={avg_fm:.3f} "
                    f"D={avg_d:.3f}"
                )
                sw.add_scalar('train/loss_g', avg_g, steps)
                sw.add_scalar('train/loss_mel', avg_mel, steps)
                sw.add_scalar('train/loss_d', avg_d, steps)

            accum_loss_g = 0.0
            accum_loss_d = 0.0
            accum_loss_mel = 0.0
            accum_loss_fm = 0.0

            # ── Checkpoint (main process only) ──────────────────────────────
            if is_main_process(rank) and steps % args.save_interval == 0 and steps > 0:
                ckpt_path = Path(args.checkpoint_dir) / f'ckpt_{steps:06d}.pt'
                save_checkpoint(ckpt_path, vocoder, mpd, mrd, optim_g, optim_d, steps, epoch, args.phase, args.projection)
                log(f"[Train] Saved checkpoint: {ckpt_path}")
                prune_old_checkpoints(Path(args.checkpoint_dir), args.keep_last_checkpoints)

            # ── Validation (main process only) ─────────────────────────────
            if (val_loader is not None
                    and is_main_process(rank)
                    and steps % args.val_interval == 0
                    and steps > 0):
                val_mel = validate(vocoder, mel_loss_fn, val_loader, device, sw, steps)
                log(f"[Val] Step {steps:,d} | val_mel={val_mel:.3f}")

            steps += 1
            if steps >= args.steps:
                if is_main_process(rank):
                    log(f"[Train] Reached {args.steps:,d} steps. Done.")
                    ckpt_path = Path(args.checkpoint_dir) / f'ckpt_{steps:06d}_final.pt'
                    save_checkpoint(ckpt_path, vocoder, mpd, mrd, optim_g, optim_d, steps, epoch, args.phase, args.projection)
                cleanup_distributed()
                return

        # Discard leftover micro-batches that didn't complete a full
        # accumulation window — partial averages would skew the update.
        if micro % accum_steps != 0:
            optim_d.zero_grad()
            optim_g.zero_grad()

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
    parser.add_argument('--projection', default='linear',
                        choices=['linear', 'mlp', 'conv', 'conv_bn', 'deconv'],
                        help='Projection layer type: linear (baseline), mlp (2-layer), '
                             'conv (temporal conv1d), conv_bn (conv + batchnorm), '
                             'deconv (learned upsampling via transposed conv)')
    parser.add_argument('--phase', choices=['A', 'B'], default='A',
                        help='A=projection only, B=end-to-end fine-tune')
    parser.add_argument('--lr_g', type=float, default=None,
                        help='Generator learning rate. Default: 2e-4 for Phase A, 1e-4 for Phase B.')
    parser.add_argument('--lr_d', type=float, default=None,
                        help='Discriminator learning rate. Default: same as generator LR.')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--segment_seconds', type=int, default=1,
                        help='Audio segment length in seconds for training crops')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--keep_last_checkpoints', type=int, default=3,
                        help='Number of periodic checkpoints to keep on disk. '
                             'Older ones are pruned. 0 = keep all. Final checkpoint is always kept.')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Number of micro-batches to accumulate before each optimizer step. '
                             'Effective batch size = batch_size * num_gpus * accumulation_steps.')
    parser.add_argument('--grad_clip', type=float, default=1000.0,
                        help='Gradient norm clipping threshold. 0 = disabled. '
                             'Default 1000 matches BigVGAN reference config.')
    parser.add_argument('--val_audio_dir', default=None,
                        help='Root directory of validation .wav files')
    parser.add_argument('--val_feat_dir', default=None,
                        help='Root directory of validation .pt WavLM feature files')
    parser.add_argument('--val_interval', type=int, default=5000,
                        help='Run validation every N steps (default: 5000)')
    parser.add_argument('--supplementary_dirs', nargs='*', metavar='AUDIO:FEAT',
                        help='Additional data directories as audio_dir:feat_dir pairs. '
                             'These are oversampled to the target weight. '
                             'Example: --supplementary_dirs /data/vocalsound:/data/vocalsound-feats '
                             '/data/emovdb:/data/emovdb-feats')
    parser.add_argument('--supplementary_weight', type=float, default=0.15,
                        help='Target fraction of supplementary data per epoch (default: 0.15 = 15%%).')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
