"""
Batch WavLM feature extraction for BigVGAN fine-tuning.

Walks an audio directory, extracts WavLM layer-6 features for each .wav file,
and saves them as .pt files in a parallel output directory tree.

With --prematch, applies kNN prematched training: for each utterance, builds a
matching pool from the same speaker's other utterances, runs kNN regression
(k=4 by default), and saves the prematched features instead. This closes the
train/inference gap described in the kNN-VC paper (Section 5.2).

Speaker grouping is determined by --speaker_depth: the number of directory levels
above the audio file that identify the speaker.
  - LibriTTS (speaker/chapter/file.wav): --speaker_depth 2
  - VCTK (speaker/file.wav): --speaker_depth 1

Usage:
  # Extract clean features (no prematching):
  python scripts/extract_wavlm_features.py \\
      --audio_dir /path/to/audio \\
      --out_dir /path/to/features \\
      [--device cuda]

  # Extract prematched features (LibriTTS layout):
  python scripts/extract_wavlm_features.py \\
      --audio_dir /path/to/audio \\
      --out_dir /path/to/features \\
      --prematch --speaker_depth 2 \\
      [--topk 4] [--device cuda]

Output:
  For each audio_dir/<subpath>/<file>.wav, writes:
    out_dir/<subpath>/<file>.pt  — shape (seq_len, 1024) float32 tensor

Features are extracted using the pretrained WavLM-Large model from kNN-VC.
Already-extracted files are skipped (resumable).
"""

import argparse
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


def load_wavlm(device):
    """Load WavLM-Large via kNN-VC hubconf."""
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))
    from hubconf import wavlm_large
    model = wavlm_large(pretrained=True, progress=True, device=str(device))
    return model


@torch.inference_mode()
def extract_features(model, wav_path: Path, device: torch.device, target_sr: int = 16000):
    """Extract WavLM layer-6 features from a wav file.

    Returns tensor of shape (seq_len, 1024).
    """
    wav, sr = torchaudio.load(wav_path, normalize=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.mean(dim=0, keepdim=True)  # mono, (1, T)
    wav = wav.to(device)

    features = model.extract_features(
        wav, output_layer=6, ret_layer_results=False
    )[0]  # (1, seq_len, 1024)
    return features.squeeze(0).cpu()  # (seq_len, 1024)


def get_speaker_id(wav_path: Path, audio_dir: Path, speaker_depth: int) -> str:
    """Extract speaker ID from the directory structure.

    speaker_depth=1: speaker/file.wav → speaker
    speaker_depth=2: speaker/chapter/file.wav → speaker
    """
    rel = wav_path.relative_to(audio_dir)
    return str(rel.parents[speaker_depth - 1]) if speaker_depth <= len(rel.parents) else rel.parts[0]


def fast_cosine_dist(source_feats, matching_pool):
    """Compute cosine distances between source frames and matching pool.

    Args:
        source_feats: (src_len, dim)
        matching_pool: (pool_len, dim)
    Returns:
        dists: (src_len, pool_len)
    """
    source_norms = torch.norm(source_feats, p=2, dim=-1)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -torch.cdist(source_feats[None], matching_pool[None], p=2)[0] ** 2 + source_norms[:, None] ** 2 + matching_norms[None] ** 2
    dotprod /= 2
    dists = 1 - (dotprod / (source_norms[:, None] * matching_norms[None] + 1e-8))
    return dists


@torch.inference_mode()
def knn_prematch(source_feats, matching_pool, topk=4, query_chunk=1024):
    """Apply kNN regression to replace source features with prematched features.

    For each frame in source_feats, find the top-k nearest neighbors in matching_pool
    by cosine distance and average them. Computed on whatever device the inputs
    live on; chunk the query dim to cap peak memory on large pools.

    Args:
        source_feats: (src_len, 1024)
        matching_pool: (pool_len, 1024)
        topk: number of neighbors
        query_chunk: max number of query frames to process at once

    Returns:
        prematched: (src_len, 1024) on the same device as inputs
    """
    outputs = []
    for start in range(0, source_feats.shape[0], query_chunk):
        chunk = source_feats[start:start + query_chunk]
        dists = fast_cosine_dist(chunk, matching_pool)      # (chunk, pool_len)
        best = dists.topk(k=topk, dim=-1, largest=False)    # (chunk, topk)
        outputs.append(matching_pool[best.indices].mean(dim=1))
    return torch.cat(outputs, dim=0)


def fmt_eta(seconds):
    if seconds < 0:
        return "?"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"


def extract_all(args):
    """Extract features for all files, without prematching."""
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)

    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(audio_dir.rglob(f'*.{args.ext}'))
    print(f"Found {len(audio_files):,d} .{args.ext} files in {audio_dir}")

    if len(audio_files) == 0:
        print("No audio files found. Check --audio_dir.")
        return

    print(f"Loading WavLM-Large on {device}...")
    model = load_wavlm(device)

    skipped = 0
    processed = 0
    errors = 0
    t0 = time.monotonic()

    for i, wav_path in enumerate(audio_files):
        rel_path = wav_path.relative_to(audio_dir).with_suffix('.pt')
        out_path = out_dir / rel_path

        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            feats = extract_features(model, wav_path, device)
            torch.save(feats.half(), out_path)
            processed += 1

            if (processed + skipped) % 100 == 0 or i == len(audio_files) - 1:
                elapsed = time.monotonic() - t0
                done = i + 1
                remaining = len(audio_files) - done
                rate = done / elapsed if elapsed > 0 else 0
                eta = remaining / rate if rate > 0 else -1
                print(
                    f"[{done}/{len(audio_files)}] "
                    f"processed={processed} skipped={skipped} errors={errors} | "
                    f"{wav_path.name} → {tuple(feats.shape)} | "
                    f"{rate:.1f} files/s, ETA {fmt_eta(eta)}"
                )
        except Exception as e:
            errors += 1
            print(f"ERROR processing {wav_path}: {e}")

    elapsed = time.monotonic() - t0
    print(f"\nDone in {fmt_eta(elapsed)}. Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    print(f"Features saved to: {out_dir}")


def extract_prematched(args):
    """Extract features with kNN prematched training.

    Two-pass approach:
      1. Extract and cache WavLM features for all utterances (or load from existing .pt)
      2. For each speaker, build matching pool from other utterances, run kNN regression
    """
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)

    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(audio_dir.rglob(f'*.{args.ext}'))
    print(f"Found {len(audio_files):,d} .{args.ext} files in {audio_dir}")

    if len(audio_files) == 0:
        print("No audio files found. Check --audio_dir.")
        return

    # Group files by speaker
    speaker_files = defaultdict(list)
    for wav_path in audio_files:
        spk = get_speaker_id(wav_path, audio_dir, args.speaker_depth)
        speaker_files[spk].append(wav_path)

    print(f"Found {len(speaker_files)} speakers (speaker_depth={args.speaker_depth})")
    for spk in sorted(speaker_files)[:5]:
        print(f"  {spk}: {len(speaker_files[spk])} utterances")
    if len(speaker_files) > 5:
        print(f"  ... and {len(speaker_files) - 5} more")

    # Check how many speakers have only 1 utterance (prematching needs >=2)
    single_utt_spks = [s for s, files in speaker_files.items() if len(files) < 2]
    if single_utt_spks:
        print(f"WARNING: {len(single_utt_spks)} speaker(s) have only 1 utterance — "
              f"these will be saved without prematching.")

    print(f"Loading WavLM-Large on {device}...")
    model = load_wavlm(device)

    total_processed = 0
    total_skipped = 0
    total_errors = 0
    n_speakers = len(speaker_files)
    t0 = time.monotonic()
    speakers_done = 0

    for spk_idx, (spk, files) in enumerate(sorted(speaker_files.items())):
        # Check if all outputs for this speaker already exist
        all_exist = all(
            (out_dir / f.relative_to(audio_dir).with_suffix('.pt')).exists()
            for f in files
        )
        if args.skip_existing and all_exist:
            total_skipped += len(files)
            speakers_done += 1
            continue

        elapsed = time.monotonic() - t0
        if speakers_done > 0:
            rate = speakers_done / elapsed
            eta = (n_speakers - spk_idx) / rate
            eta_str = f" | ETA {fmt_eta(eta)}"
        else:
            eta_str = ""
        print(f"\n[Speaker {spk_idx+1}/{n_speakers}] {spk} — {len(files)} utterances{eta_str}")

        # Pass 1: Extract raw features for this speaker (stored as fp16 on CPU
        # to halve memory — upcast to fp32 when moving to GPU for kNN).
        raw_feats = {}  # wav_path -> (seq_len, 1024) fp16 CPU tensor
        t_spk = time.monotonic()
        for fi, wav_path in enumerate(files):
            try:
                feats = extract_features(model, wav_path, device)
                raw_feats[wav_path] = feats.half()
                if (fi + 1) % 50 == 0 or fi == len(files) - 1:
                    spk_elapsed = time.monotonic() - t_spk
                    spk_rate = (fi + 1) / spk_elapsed if spk_elapsed > 0 else 0
                    spk_eta = (len(files) - fi - 1) / spk_rate if spk_rate > 0 else -1
                    print(f"  extract [{fi+1}/{len(files)}] {spk_rate:.1f} files/s, ETA {fmt_eta(spk_eta)}")
            except Exception as e:
                total_errors += 1
                print(f"  ERROR extracting {wav_path.name}: {e}")

        if len(raw_feats) < 2:
            # Can't prematch with fewer than 2 utterances, save raw
            for wav_path, feats in raw_feats.items():
                out_path = out_dir / wav_path.relative_to(audio_dir).with_suffix('.pt')
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feats.half(), out_path)
                total_processed += 1
            continue

        # Build the full speaker pool once (on GPU if available) and mask out
        # the current utterance's frames when prematching each query. This avoids
        # re-concatenating a large tensor for every utterance.
        all_paths = list(raw_feats.keys())
        offsets = [0]
        for p in all_paths:
            offsets.append(offsets[-1] + raw_feats[p].shape[0])
        full_pool = torch.cat([raw_feats[p].float() for p in all_paths], dim=0).to(device)  # (N_total, 1024)
        pool_len = full_pool.shape[0]

        # Pass 2: For each utterance, prematch against all other utterances from same speaker
        t_pm = time.monotonic()
        pm_done = 0
        for idx, wav_path in enumerate(all_paths):
            out_path = out_dir / wav_path.relative_to(audio_dir).with_suffix('.pt')

            if args.skip_existing and out_path.exists():
                total_skipped += 1
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Exclude this utterance's frames from the pool by slicing around them
            start, end = offsets[idx], offsets[idx + 1]
            if start == 0:
                matching_pool = full_pool[end:]
            elif end == pool_len:
                matching_pool = full_pool[:start]
            else:
                matching_pool = torch.cat([full_pool[:start], full_pool[end:]], dim=0)

            source_feats = raw_feats[wav_path].float().to(device)
            prematched = knn_prematch(source_feats, matching_pool, topk=args.topk)
            # Save as float16 to halve disk usage; dataset upcasts to float32 at load.
            torch.save(prematched.half().cpu(), out_path)
            total_processed += 1
            pm_done += 1
            if pm_done % 50 == 0 or idx == len(all_paths) - 1:
                pm_elapsed = time.monotonic() - t_pm
                pm_rate = pm_done / pm_elapsed if pm_elapsed > 0 else 0
                pm_remaining = len(all_paths) - idx - 1
                pm_eta = pm_remaining / pm_rate if pm_rate > 0 else -1
                print(f"  prematch [{idx+1}/{len(all_paths)}] {pm_rate:.1f} files/s, ETA {fmt_eta(pm_eta)}")

        speakers_done += 1
        print(f"  Done. Pool size: {pool_len:,d} frames")

        # Free memory between speakers
        del raw_feats, full_pool
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.monotonic() - t0
    print(f"\nDone in {fmt_eta(elapsed)}. Processed: {total_processed}, Skipped: {total_skipped}, Errors: {total_errors}")
    print(f"Prematched features saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Extract WavLM features for BigVGAN training')
    parser.add_argument('--audio_dir', required=True, help='Root directory containing .wav files')
    parser.add_argument('--out_dir', required=True, help='Output directory for .pt feature files')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ext', default='wav', help='Audio file extension (default: wav)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip files that already have features (default: True)')
    parser.add_argument('--prematch', action='store_true',
                        help='Apply kNN prematched training (recommended for vocoder quality)')
    parser.add_argument('--topk', type=int, default=4,
                        help='Number of kNN neighbors for prematching (default: 4)')
    parser.add_argument('--speaker_depth', type=int, default=2,
                        help='Directory depth for speaker ID. '
                             '2 for LibriTTS (speaker/chapter/file.wav), '
                             '1 for VCTK (speaker/file.wav). Default: 2')
    args = parser.parse_args()

    if args.prematch:
        extract_prematched(args)
    else:
        extract_all(args)


if __name__ == '__main__':
    main()
