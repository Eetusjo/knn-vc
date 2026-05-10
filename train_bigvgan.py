"""
BigVGAN fine-tuning script for kNN-VC.

Two modes:

  --mode projection (default): Mel-input BigVGAN-v2 (24 kHz, 100-band) with a
  learned WavLM→mel projection module bolted in front. Trained in two phases:
    Phase A (~50k steps): Train projection layer only, BigVGAN backbone frozen.
    Phase B (~100k-300k steps): Unfreeze BigVGAN, train end-to-end at lower LR.

  --mode direct: BigVGAN configured to consume WavLM features directly
  (num_mels=1024 input dim, hop_size=480, modified upsample stack). No
  projection, no F.interpolate. Single-phase end-to-end training. By default
  warm-starts the transferable layers (resblocks + conv_post + unchanged
  upsamplers) from NVIDIA's pretrained 100-band 24 kHz checkpoint; conv_pre
  and the changed upsamplers train from random init. Recommended:
  --gan_warmup_steps 10000–20000 to let the input adapter find scale on the
  mel target before the discriminators get involved.

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
import json
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
from bigvgan_vocoder import (
    BigVGANVocoder,
    BigVGANDirectVocoder,
    build_projection,
    load_bigvgan_partial,
)


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
# Bandwidth-aware mel loss helper
# ──────────────────────────────────────────────────────────────────────────────

def _bandlimit_to(wav: torch.Tensor, src_sr: int, native_sr: int) -> torch.Tensor:
    """Round-trip resample wav (..., T) through native_sr to remove content above
    native_sr/2. No-op when native_sr >= src_sr."""
    if native_sr >= src_sr:
        return wav
    T = wav.shape[-1]
    x = torchaudio.functional.resample(wav, src_sr, native_sr)
    x = torchaudio.functional.resample(x, native_sr, src_sr)
    if x.shape[-1] > T:
        x = x[..., :T]
    elif x.shape[-1] < T:
        x = F.pad(x, (0, T - x.shape[-1]))
    return x


def _bandlimit_batch(wav_real: torch.Tensor, wav_gen: torch.Tensor,
                     native_srs: torch.Tensor, src_sr: int):
    """For samples whose native SR is below src_sr, low-pass both the real and
    generated waveforms to that SR. Returns the (possibly modified) tensors;
    if every sample is wideband the inputs are returned unchanged."""
    if bool((native_srs >= src_sr).all().item()):
        return wav_real, wav_gen
    real_chunks = []
    gen_chunks = []
    for i in range(wav_real.shape[0]):
        ns = int(native_srs[i].item())
        if ns >= src_sr:
            real_chunks.append(wav_real[i:i + 1])
            gen_chunks.append(wav_gen[i:i + 1])
        else:
            real_chunks.append(_bandlimit_to(wav_real[i:i + 1], src_sr, ns))
            gen_chunks.append(_bandlimit_to(wav_gen[i:i + 1], src_sr, ns))
    return torch.cat(real_chunks, dim=0), torch.cat(gen_chunks, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class WavLMVocoderDataset(Dataset):
    """Paired (WavLM features, audio waveform, native_sr) dataset for BigVGAN fine-tuning.

    Expects one or more (audio_dir, feat_dir) pairs. Each feat_dir contains .pt files
    (WavLM layer-6 features, shape (seq_len, 1024)) with the same relative paths as
    the corresponding audio files but with .pt extension.

    Segment size is in samples at target_sr. Features are trimmed/padded to match.

    When supplementary directories are provided, `group_of` tracks which samples
    belong to the primary (0) vs supplementary (1) group for weighted sampling.
    `native_sr_of` records the source SR per sample so the training loop can
    band-limit narrowband clips before computing the mel loss.
    """

    AUDIO_EXTS = ('.wav', '.flac', '.mp3', '.ogg', '.opus')

    def __init__(
        self,
        audio_dir: Path,
        feat_dir: Path,
        supplementary_dirs: list[tuple[Path, Path, int]] | list[tuple[Path, Path]] | None = None,
        segment_size: int = 24000 * 1,  # 1 second at 24kHz
        target_sr: int = 24000,
        wavlm_frame_rate: int = 50,  # 50 frames/sec = 20ms hop
        split: bool = True,
        verbose: bool = True,
        primary_native_sr: int = 24000,
    ):
        self.segment_size = segment_size
        self.target_sr = target_sr
        self.wavlm_frame_rate = wavlm_frame_rate
        self.split = split
        self.frames_per_seg = math.ceil(segment_size / target_sr * wavlm_frame_rate)
        self._verbose = verbose

        self.pairs = []
        self.group_of = []        # 0 = primary, 1 = supplementary
        self.native_sr_of = []    # source SR per sample (for bandwidth-aware loss)

        self._scan_dirs(Path(audio_dir), Path(feat_dir), group=0,
                        native_sr=int(primary_native_sr))
        for sup in (supplementary_dirs or []):
            if len(sup) == 3:
                sup_audio, sup_feat, sup_sr = sup
            else:
                sup_audio, sup_feat = sup
                sup_sr = target_sr
            self._scan_dirs(Path(sup_audio), Path(sup_feat), group=1,
                            native_sr=int(sup_sr))

        if len(self.pairs) == 0:
            raise ValueError(
                f"No paired (audio, feature) files found.\n"
                f"  audio_dir: {audio_dir}\n"
                f"  feat_dir: {feat_dir}\n"
                "Run scripts/extract_wavlm_features.py first."
            )

        n_primary = self.group_of.count(0)
        n_supp = self.group_of.count(1)
        if self._verbose:
            log(f"[Dataset] {n_primary:,d} primary + {n_supp:,d} supplementary = {len(self.pairs):,d} total pairs.")
            sr_counts = {}
            for sr in self.native_sr_of:
                sr_counts[sr] = sr_counts.get(sr, 0) + 1
            sr_summary = ', '.join(f"{sr} Hz: {n:,d}" for sr, n in sorted(sr_counts.items()))
            log(f"[Dataset] Native-SR breakdown — {sr_summary}")

    def _scan_dirs(self, audio_dir: Path, feat_dir: Path, group: int, native_sr: int):
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
                self.native_sr_of.append(native_sr)
                found += 1
            else:
                missing_audio += 1
        if self._verbose:
            if missing_audio:
                log(f"[Dataset] WARNING: {missing_audio:,d} .pt files have no matching audio in {audio_dir}.")
            label = "primary" if group == 0 else "supplementary"
            log(f"[Dataset] Found {found:,d} paired files in {feat_dir} ({label}, native_sr={native_sr}).")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio_path, feat_path = self.pairs[idx]
        native_sr = self.native_sr_of[idx]

        try:
            wav, sr = torchaudio.load(audio_path, normalize=True)
            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            wav = wav.mean(dim=0)  # mono, shape (T,)
            feats = torch.load(feat_path, map_location='cpu').float()  # (seq_len, 1024)
        except (RuntimeError, Exception) as e:
            log(f"[Dataset] Skipping corrupt file {feat_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.pairs) - 1))

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

        return feats, wav, native_sr  # (frames_per_seg, 1024), (segment_size,), int


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
                    projection_type='linear', mode='projection'):
    # Unwrap DDP modules to save the underlying state_dict
    def unwrap(m):
        return m.module if isinstance(m, DDP) else m

    state = {
        'mode': mode,
        'bigvgan': unwrap(vocoder).bigvgan.state_dict(),
        'optim_g': optim_g.state_dict(),
        'steps': steps,
        'epoch': epoch,
        'phase': phase,
    }
    if mode == 'projection':
        state['projection'] = unwrap(vocoder).projection.state_dict()
        state['projection_type'] = projection_type
    # Phase A skips discriminators entirely; their state is only present in Phase B.
    if mpd is not None:
        state['mpd'] = unwrap(mpd).state_dict()
    if mrd is not None:
        state['mrd'] = unwrap(mrd).state_dict()
    if optim_d is not None:
        state['optim_d'] = optim_d.state_dict()

    # Atomic write: save to .tmp then rename, so an interrupted save doesn't leave
    # a half-written checkpoint on disk.
    tmp_path = str(path) + '.tmp'
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location='cpu')


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
def validate(vocoder, mel_loss_fn, val_loader, device, sw, steps,
             rank, world_size, src_sr, num_audio_samples=4):
    """Run validation across all ranks, then all-reduce the loss."""
    model = vocoder.module if hasattr(vocoder, 'module') else vocoder
    model.eval()

    total_mel = 0.0
    n_batches = 0
    audio_logged = 0

    for feats, wav_real, native_srs in val_loader:
        feats = feats.to(device)
        wav_real = wav_real.to(device).unsqueeze(1)
        native_srs = native_srs.to(device)

        wav_gen = model(feats)

        min_len = min(wav_real.shape[-1], wav_gen.shape[-1])
        wav_real = wav_real[..., :min_len]
        wav_gen = wav_gen[..., :min_len]

        # Apply per-sample band-limit so val mel loss matches the training-time loss
        # for any narrowband validation samples (typically a no-op).
        wav_real_for_mel, wav_gen_for_mel = _bandlimit_batch(
            wav_real, wav_gen, native_srs, src_sr=src_sr)
        total_mel += mel_loss_fn(wav_gen_for_mel, wav_real_for_mel).item()
        n_batches += 1

        if sw is not None and audio_logged < num_audio_samples:
            for j in range(min(feats.shape[0], num_audio_samples - audio_logged)):
                sw.add_audio(f'val/gen_{audio_logged}', wav_gen[j], steps, sample_rate=24000)
                sw.add_audio(f'val/real_{audio_logged}', wav_real[j], steps, sample_rate=24000)
                audio_logged += 1

    if world_size > 1:
        stats = torch.tensor([total_mel, n_batches], dtype=torch.float64, device=device)
        dist.all_reduce(stats)
        total_mel = stats[0].item()
        n_batches = int(stats[1].item())

    val_mel = total_mel / max(n_batches, 1)

    if sw is not None:
        sw.add_scalar('val/loss_mel', val_mel, steps)

    model.train()
    return val_mel


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if is_main_process(rank):
        log(f"[Train] world_size={world_size}, device={device} mode={args.mode}")

    # ── Build BigVGAN + vocoder wrapper ──────────────────────────────────────
    if args.mode == 'projection':
        # Mel-input BigVGAN with a separate WavLM→mel projection module.
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
        h = bigvgan_model.h  # sampling_rate=24000, num_mels=100, hop_size=256, ...

        projection = build_projection(args.projection, in_dim=1024, out_dim=h.num_mels)
        vocoder = BigVGANVocoder(bigvgan_model, projection, target_sr=h.sampling_rate).to(device)
        if is_main_process(rank):
            n_proj = sum(p.numel() for p in projection.parameters())
            log(f"[Train] Projection: {args.projection} ({n_proj:,d} params)")
    else:
        # Direct path: BigVGAN consumes WavLM features end-to-end (num_mels=1024,
        # hop_size=480, modified upsample stack). Optionally warm-start the
        # transferable layers (resblocks + conv_post + unchanged upsamplers)
        # from NVIDIA's pretrained 100-band 24 kHz checkpoint.
        from bigvgan.env import AttrDict as BVGAttrDict
        if args.direct_config:
            cfg_path = Path(args.direct_config)
        else:
            import bigvgan as bigvgan_pkg
            cfg_path = (Path(bigvgan_pkg.__file__).parent / 'configs'
                        / 'bigvgan_v2_24khz_wavlm_480x.json')
        with open(cfg_path) as f:
            h = BVGAttrDict(json.loads(f.read()))
        bigvgan_model = bigvgan_module.BigVGAN(h).to(device)
        if args.warm_start_from_pretrained and not args.resume:
            if is_main_process(rank):
                log(f"[Train] Warm-starting direct BigVGAN from pretrained 100-band weights.")
            pretrained = bigvgan_module.BigVGAN._from_pretrained(
                model_id='nvidia/bigvgan_v2_24khz_100band_256x',
                revision=None, cache_dir=None, force_download=False,
                proxies=None, resume_download=False, local_files_only=False,
                token=None, use_cuda_kernel=False, map_location=str(device),
            )
            load_bigvgan_partial(bigvgan_model, pretrained.state_dict(),
                                 verbose=is_main_process(rank))
            del pretrained

        vocoder = BigVGANDirectVocoder(bigvgan_model, target_sr=h.sampling_rate).to(device)
        if is_main_process(rank):
            log(f"[Train] Direct mode: hop_size={h.hop_size}, sr={h.sampling_rate}, "
                f"upsample_rates={list(h.upsample_rates)}.")

    # In projection mode, Phase A trains the projection on mel-loss only — the
    # discriminators would be random-init noise that fights the projection. In
    # direct mode there's no Phase A, but the input adapter (conv_pre + the
    # changed upsamplers) is also random-init, so we honor --gan_warmup_steps
    # to delay discriminator updates similarly.
    use_disc = (args.mode == 'direct') or (args.phase == 'B')

    # Resolve grad-clip and mel-loss weight from config when not set on CLI.
    grad_clip = args.grad_clip if args.grad_clip is not None else h.get('clip_grad_norm', 1000.0)
    lambda_mel = float(h.get('lambda_melloss', 45.0))
    if is_main_process(rank):
        log(f"[Train] grad_clip={grad_clip:g}, lambda_melloss={lambda_mel:g}")

    # ── Discriminators (Phase B only; match BigVGAN-v2 config) ───────────────
    mpd = None
    mrd = None
    if use_disc:
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
        ckpt_mode = ckpt.get('mode', 'projection')
        if ckpt_mode != args.mode:
            raise ValueError(
                f"Checkpoint trained with mode={ckpt_mode!r} but --mode={args.mode!r}. "
                f"Use --mode {ckpt_mode} to match the checkpoint.")
        if args.mode == 'projection':
            ckpt_proj_type = ckpt.get('projection_type', 'linear')
            if ckpt_proj_type != args.projection:
                raise ValueError(
                    f"Checkpoint uses projection '{ckpt_proj_type}' but "
                    f"--projection={args.projection}. Use --projection "
                    f"{ckpt_proj_type} to match the checkpoint.")
            vocoder.projection.load_state_dict(ckpt['projection'])
        vocoder.bigvgan.load_state_dict(ckpt['bigvgan'])
        # Phase-A checkpoints don't store discriminator state; let them init fresh
        # for the Phase A→B transition.
        if mpd is not None and 'mpd' in ckpt:
            mpd.load_state_dict(ckpt['mpd'])
        if mrd is not None and 'mrd' in ckpt:
            mrd.load_state_dict(ckpt['mrd'])
        steps = ckpt.get('steps', 0)
        start_epoch = ckpt.get('epoch', 0)
        resumed_phase = ckpt.get('phase', None)
        if is_main_process(rank):
            log(f"[Train] Resumed from {args.resume} at step {steps} "
                f"(mode={ckpt_mode}, phase={resumed_phase})")

    # ── Freeze/unfreeze, parameter group, default LR ─────────────────────────
    if args.mode == 'direct':
        # Direct mode: end-to-end from the start. No Phase A/B.
        if is_main_process(rank):
            log("[Train] Direct mode: end-to-end training (no projection, no phase split).")
        for p in vocoder.bigvgan.parameters():
            p.requires_grad = True
        g_params = list(vocoder.bigvgan.parameters())
        default_lr = 1e-4
    elif args.phase == 'A':
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
        vocoder = DDP(vocoder, device_ids=[local_rank], find_unused_parameters=False)
        if use_disc:
            mpd = DDP(mpd, device_ids=[local_rank])
            mrd = DDP(mrd, device_ids=[local_rank])

    # ── Optimizers ───────────────────────────────────────────────────────────
    optim_g = torch.optim.AdamW(g_params, lr=lr_g, betas=(0.8, 0.99))
    optim_d = None
    if use_disc:
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
            if optim_d is not None and 'optim_d' in ckpt:
                optim_d.load_state_dict(ckpt['optim_d'])
        else:
            if is_main_process(rank):
                log(f"[Train] Phase change (ckpt={resumed_phase} → run={args.phase}): "
                    f"re-initializing optimizer states with fresh LR={lr_g:.1e}.")

    if ckpt is not None:
        del ckpt
        torch.cuda.empty_cache()

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.999, last_epoch=max(-1, start_epoch - 1))
    scheduler_d = None
    if use_disc:
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.999, last_epoch=max(-1, start_epoch - 1))

    # ── Loss functions ────────────────────────────────────────────────────────
    mel_loss_fn = MultiScaleMelSpectrogramLoss(sampling_rate=h.sampling_rate).to(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    segment_size = h.sampling_rate * args.segment_seconds  # samples at 24kHz

    supplementary_dirs = None
    if args.supplementary_dirs:
        supplementary_dirs = []
        for spec in args.supplementary_dirs:
            parts = spec.split(':')
            if len(parts) == 2:
                a_dir, f_dir = parts
                native_sr = h.sampling_rate
            elif len(parts) == 3:
                a_dir, f_dir, native_sr_str = parts
                native_sr = int(native_sr_str)
            else:
                raise ValueError(
                    f"Invalid --supplementary_dirs entry '{spec}'. "
                    f"Expected AUDIO:FEAT or AUDIO:FEAT:NATIVE_SR.")
            supplementary_dirs.append((Path(a_dir), Path(f_dir), native_sr))

    dataset = WavLMVocoderDataset(
        audio_dir=args.audio_dir,
        feat_dir=args.feat_dir,
        supplementary_dirs=supplementary_dirs,
        segment_size=segment_size,
        target_sr=h.sampling_rate,
        verbose=is_main_process(rank),
    )

    has_supplementary = dataset.group_of.count(1) > 0
    if has_supplementary and args.supplementary_weight > 0:
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
            verbose=is_main_process(rank),
        )
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
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
    if use_disc:
        mpd.train()
        mrd.train()

    accum_steps = args.accumulation_steps
    use_ddp = world_size > 1

    # no_sync() skips DDP all-reduce on intermediate micro-batches
    def maybe_no_sync(model, is_last):
        if use_ddp and not is_last:
            return model.no_sync()
        return contextlib.nullcontext()

    if use_disc:
        optim_d.zero_grad()
    optim_g.zero_grad()

    for epoch in range(start_epoch, 10000):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        elif has_supplementary and args.supplementary_weight > 0:
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

        for feats, wav_real, native_srs in loader:
            feats = feats.to(device)       # (B, frames, 1024)
            wav_real = wav_real.to(device).unsqueeze(1)  # (B, 1, T)
            native_srs = native_srs.to(device)            # (B,) int64
            micro += 1
            is_last_micro = (micro % accum_steps == 0)

            # Forward pass: WavLM features → synthesized audio (at 24kHz, no resample)
            wav_gen = vocoder(feats)  # (B, 1, T')

            # Align lengths (resampling may cause off-by-one)
            min_len = min(wav_real.shape[-1], wav_gen.shape[-1])
            wav_real = wav_real[..., :min_len]
            wav_gen = wav_gen[..., :min_len]

            # Bandwidth-aware mel target: low-pass narrowband samples so we don't
            # penalize the generator for producing energy above the source's Nyquist.
            wav_real_for_mel, wav_gen_for_mel = _bandlimit_batch(
                wav_real, wav_gen, native_srs, src_sr=h.sampling_rate)

            # Wideband subset is the only one that participates in the GAN path —
            # otherwise D would learn "wideband ⇒ fake" for the band-limited clips.
            wb_mask = native_srs >= h.sampling_rate
            has_wb = bool(wb_mask.any().item())
            wb_idx = torch.where(wb_mask)[0] if has_wb else None

            loss_d = None
            loss_fm_mpd = None
            loss_fm_mrd = None

            # Direct-mode warmup: skip GAN losses entirely while the random-init
            # input adapter (conv_pre + 3 changed upsamplers) finds reasonable
            # scale on the mel target. Mirrors the projection-mode Phase-A logic.
            gan_active = use_disc and (steps >= args.gan_warmup_steps)

            if gan_active and has_wb:
                wb_real = wav_real[wb_idx]
                wb_gen = wav_gen[wb_idx]

                # ── Discriminator update (per-micro, mirrors reference) ─────
                # zero_grad → D backward → step D. The G-backward below will
                # write polluting grads onto D's params (FM and gen-GAN terms
                # have non-zero gradient w.r.t. D's params), but they're never
                # applied — the next micro's optim_d.zero_grad() wipes them
                # before the next D backward.
                #
                # D updates every micro-batch on a single-batch's worth of
                # gradient (no accumulation), while G accumulates over
                # accum_steps. This is the natural translation of the BigVGAN
                # reference's pattern to gradient-accumulation training. D
                # tends to benefit from frequent updates with smaller batches.
                optim_d.zero_grad()

                y_df_r, y_df_g, _, _ = mpd(wb_real, wb_gen.detach())
                loss_d_mpd, _, _ = discriminator_loss(y_df_r, y_df_g)

                y_dr_r, y_dr_g, _, _ = mrd(wb_real, wb_gen.detach())
                loss_d_mrd, _, _ = discriminator_loss(y_dr_r, y_dr_g)

                loss_d = loss_d_mpd + loss_d_mrd
                loss_d.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        itertools.chain(mpd.parameters(), mrd.parameters()),
                        grad_clip,
                    )
                optim_d.step()

            # ── Generator update ────────────────────────────────────────────
            with contextlib.ExitStack() as stack:
                stack.enter_context(maybe_no_sync(vocoder, is_last_micro))

                loss_mel = mel_loss_fn(wav_gen_for_mel, wav_real_for_mel)

                if gan_active and has_wb:
                    y_df_r, y_df_g, fmap_f_r, fmap_f_g = mpd(wb_real, wb_gen)
                    y_dr_r, y_dr_g, fmap_r_r, fmap_r_g = mrd(wb_real, wb_gen)

                    loss_fm_mpd = feature_loss(fmap_f_r, fmap_f_g)
                    loss_fm_mrd = feature_loss(fmap_r_r, fmap_r_g)
                    loss_gen_mpd, _ = generator_loss(y_df_g)
                    loss_gen_mrd, _ = generator_loss(y_dr_g)

                    loss_g = (loss_mel * lambda_mel + loss_fm_mpd + loss_fm_mrd + loss_gen_mpd + loss_gen_mrd) / accum_steps
                else:
                    # Phase A, or Phase B with an all-narrowband micro-batch.
                    loss_g = (loss_mel * lambda_mel) / accum_steps

                loss_g.backward()

            if loss_d is not None:
                accum_loss_d += loss_d.item()
            if loss_fm_mpd is not None:
                accum_loss_fm += (loss_fm_mpd.item() + loss_fm_mrd.item())
            accum_loss_g += loss_g.item() * accum_steps
            accum_loss_mel += loss_mel.item()

            # Stop training immediately if we hit a non-finite loss — for long
            # runs this saves hours of wasted compute chasing NaNs.
            if not torch.isfinite(loss_g) or (loss_d is not None and not torch.isfinite(loss_d)):
                if is_main_process(rank):
                    d_str = f", D={loss_d.item()}" if loss_d is not None else ""
                    log(f"[Train] ABORT: non-finite loss at step {steps} "
                        f"(G={loss_g.item()}{d_str})")
                cleanup_distributed()
                return

            if not is_last_micro:
                continue

            # ── Optimizer step (G only; D was stepped per micro) ───────────
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(g_params, grad_clip)
            optim_g.step()
            optim_g.zero_grad()

            # ── Logging (main process only) ─────────────────────────────────
            if is_main_process(rank) and steps % args.log_interval == 0:
                avg_mel = accum_loss_mel / accum_steps
                avg_g = accum_loss_g / accum_steps
                if use_disc:
                    avg_fm = accum_loss_fm / accum_steps
                    avg_d = accum_loss_d / accum_steps
                    log(
                        f"Step {steps:,d} | "
                        f"G={avg_g:.3f} mel={avg_mel:.3f} "
                        f"fm={avg_fm:.3f} "
                        f"D={avg_d:.3f}"
                    )
                    sw.add_scalar('train/loss_d', avg_d, steps)
                else:
                    log(f"Step {steps:,d} | G={avg_g:.3f} mel={avg_mel:.3f}")
                sw.add_scalar('train/loss_g', avg_g, steps)
                sw.add_scalar('train/loss_mel', avg_mel, steps)

            accum_loss_g = 0.0
            accum_loss_d = 0.0
            accum_loss_mel = 0.0
            accum_loss_fm = 0.0

            # ── Checkpoint (main process only) ──────────────────────────────
            if is_main_process(rank) and steps % args.save_interval == 0 and steps > 0:
                ckpt_path = Path(args.checkpoint_dir) / f'ckpt_{steps:06d}.pt'
                save_checkpoint(ckpt_path, vocoder, mpd, mrd, optim_g, optim_d,
                                steps, epoch, args.phase, args.projection,
                                mode=args.mode)
                log(f"[Train] Saved checkpoint: {ckpt_path}")
                prune_old_checkpoints(Path(args.checkpoint_dir), args.keep_last_checkpoints)

            # ── Validation (all ranks) ─────────────────────────────────────
            if (val_loader is not None
                    and steps % args.val_interval == 0
                    and steps > 0):
                val_mel = validate(vocoder, mel_loss_fn, val_loader, device, sw, steps,
                                   rank, world_size, src_sr=h.sampling_rate)
                if is_main_process(rank):
                    log(f"[Val] Step {steps:,d} | val_mel={val_mel:.3f}")

            steps += 1
            if steps >= args.steps:
                if is_main_process(rank):
                    log(f"[Train] Reached {args.steps:,d} steps. Done.")
                    ckpt_path = Path(args.checkpoint_dir) / f'ckpt_{steps:06d}_final.pt'
                    save_checkpoint(ckpt_path, vocoder, mpd, mrd, optim_g, optim_d,
                                    steps, epoch, args.phase, args.projection,
                                    mode=args.mode)
                cleanup_distributed()
                return

        # Discard leftover micro-batches that didn't complete a full
        # accumulation window — partial averages would skew the update.
        if micro % accum_steps != 0:
            if use_disc:
                optim_d.zero_grad()
            optim_g.zero_grad()

        scheduler_g.step()
        if use_disc:
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
    parser.add_argument('--mode', choices=['projection', 'direct'], default='projection',
                        help='projection: BigVGAN consumes a learned projection of '
                             'WavLM features (mel-shaped, 100-d, 93.75 Hz). '
                             'direct: BigVGAN consumes raw WavLM features (1024-d, 50 Hz) '
                             'with hop_size=480 — no projection, no F.interpolate.')
    parser.add_argument('--projection', default='deconv',
                        choices=['linear', 'mlp', 'conv', 'conv_bn', 'deconv'],
                        help='[projection mode only] Projection layer type: linear (baseline), '
                             'mlp (2-layer), conv (temporal conv1d), conv_bn (conv + batchnorm), '
                             'deconv (learned upsampling via transposed conv, default)')
    parser.add_argument('--phase', choices=['A', 'B'], default='A',
                        help='[projection mode only] A=projection only, B=end-to-end fine-tune. '
                             'Ignored in direct mode (always end-to-end).')
    parser.add_argument('--direct_config', default=None,
                        help='[direct mode only] Path to BigVGAN config JSON. '
                             'Default: BigVGAN/configs/bigvgan_v2_24khz_wavlm_480x.json')
    parser.add_argument('--warm_start_from_pretrained', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='[direct mode only] Partial-load transferable layers from '
                             'NVIDIA bigvgan_v2_24khz_100band_256x. Default: enabled.')
    parser.add_argument('--gan_warmup_steps', type=int, default=0,
                        help='Skip discriminator updates for the first N steps (mel-loss only). '
                             'Recommended for direct mode where the input adapter is random-init '
                             '(e.g. 10000–20000). Default 0 = no warmup.')
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
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='Gradient norm clipping threshold. None (default) '
                             'reads h.clip_grad_norm from the BigVGAN config '
                             '(falls back to 1000.0 if missing). 0 = disabled.')
    parser.add_argument('--val_audio_dir', default=None,
                        help='Root directory of validation .wav files')
    parser.add_argument('--val_feat_dir', default=None,
                        help='Root directory of validation .pt WavLM feature files')
    parser.add_argument('--val_interval', type=int, default=5000,
                        help='Run validation every N steps (default: 5000)')
    parser.add_argument('--supplementary_dirs', nargs='*', metavar='AUDIO:FEAT[:NATIVE_SR]',
                        help='Additional data directories as audio_dir:feat_dir[:native_sr] '
                             'specs. These are oversampled to the target weight. '
                             'native_sr (Hz) is the source-content Nyquist for that dataset; '
                             'narrowband samples are low-passed before mel loss and skipped '
                             'from the adversarial path so band-limited content does not '
                             'dilute wideband supervision. Default native_sr = target SR. '
                             'Example: --supplementary_dirs /data/vocalsound:/data/vocalsound-feats:16000 '
                             '/data/emovdb:/data/emovdb-feats:24000')
    parser.add_argument('--supplementary_weight', type=float, default=0.15,
                        help='Target fraction of supplementary data per epoch (default: 0.15 = 15%%).')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
