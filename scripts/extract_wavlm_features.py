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
def knn_prematch(source_feats, matching_pool, topk=4):
    """Apply kNN regression to replace source features with prematched features.

    For each frame in source_feats, find the top-k nearest neighbors in matching_pool
    by cosine distance and average them.

    Args:
        source_feats: (src_len, 1024)
        matching_pool: (pool_len, 1024)
        topk: number of neighbors

    Returns:
        prematched: (src_len, 1024)
    """
    dists = fast_cosine_dist(source_feats, matching_pool)
    best = dists.topk(k=topk, dim=-1, largest=False)  # (src_len, topk)
    prematched = matching_pool[best.indices].mean(dim=1)  # (src_len, 1024)
    return prematched


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

    for i, wav_path in enumerate(audio_files):
        rel_path = wav_path.relative_to(audio_dir).with_suffix('.pt')
        out_path = out_dir / rel_path

        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            feats = extract_features(model, wav_path, device)
            torch.save(feats, out_path)
            processed += 1

            if (processed + skipped) % 100 == 0 or i == len(audio_files) - 1:
                print(
                    f"[{i+1}/{len(audio_files)}] "
                    f"processed={processed} skipped={skipped} errors={errors} | "
                    f"{wav_path.name} → shape {tuple(feats.shape)}"
                )
        except Exception as e:
            errors += 1
            print(f"ERROR processing {wav_path}: {e}")

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
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

    for spk_idx, (spk, files) in enumerate(sorted(speaker_files.items())):
        # Check if all outputs for this speaker already exist
        all_exist = all(
            (out_dir / f.relative_to(audio_dir).with_suffix('.pt')).exists()
            for f in files
        )
        if args.skip_existing and all_exist:
            total_skipped += len(files)
            continue

        print(f"\n[Speaker {spk_idx+1}/{len(speaker_files)}] {spk} — {len(files)} utterances")

        # Pass 1: Extract raw features for this speaker
        raw_feats = {}  # wav_path -> (seq_len, 1024)
        for wav_path in files:
            try:
                feats = extract_features(model, wav_path, device)
                raw_feats[wav_path] = feats
            except Exception as e:
                total_errors += 1
                print(f"  ERROR extracting {wav_path.name}: {e}")

        if len(raw_feats) < 2:
            # Can't prematch with fewer than 2 utterances, save raw
            for wav_path, feats in raw_feats.items():
                out_path = out_dir / wav_path.relative_to(audio_dir).with_suffix('.pt')
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(feats, out_path)
                total_processed += 1
            continue

        # Pass 2: For each utterance, prematch against all other utterances from same speaker
        for wav_path, source_feats in raw_feats.items():
            out_path = out_dir / wav_path.relative_to(audio_dir).with_suffix('.pt')

            if args.skip_existing and out_path.exists():
                total_skipped += 1
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)

            # Build matching pool from all other utterances of this speaker
            pool_parts = [f for p, f in raw_feats.items() if p != wav_path]
            matching_pool = torch.cat(pool_parts, dim=0)  # (pool_len, 1024)

            prematched = knn_prematch(source_feats, matching_pool, topk=args.topk)
            torch.save(prematched, out_path)
            total_processed += 1

        print(f"  Done. Pool size: {matching_pool.shape[0]:,d} frames")

        # Free memory between speakers
        del raw_feats, matching_pool
        gc.collect()

    print(f"\nDone. Processed: {total_processed}, Skipped: {total_skipped}, Errors: {total_errors}")
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
