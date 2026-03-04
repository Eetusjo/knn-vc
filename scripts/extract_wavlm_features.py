"""
Batch WavLM feature extraction for BigVGAN fine-tuning.

Walks an audio directory, extracts WavLM layer-6 features for each .wav file,
and saves them as .pt files in a parallel output directory tree.

Usage:
  python scripts/extract_wavlm_features.py \\
      --audio_dir /path/to/audio \\
      --out_dir /path/to/features \\
      [--device cuda] \\
      [--batch_size 8]

Output:
  For each audio_dir/<subpath>/<file>.wav, writes:
    out_dir/<subpath>/<file>.pt  — shape (seq_len, 1024) float32 tensor

Features are extracted using the pretrained WavLM-Large model from kNN-VC.
Already-extracted files are skipped (resumable).
"""

import argparse
import sys
from pathlib import Path

import torch
import torchaudio


def load_wavlm(device):
    """Load WavLM-Large via kNN-VC hubconf."""
    # Add parent dir to path so we can import hubconf
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


def main():
    parser = argparse.ArgumentParser(description='Extract WavLM features for BigVGAN training')
    parser.add_argument('--audio_dir', required=True, help='Root directory containing .wav files')
    parser.add_argument('--out_dir', required=True, help='Output directory for .pt feature files')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ext', default='wav', help='Audio file extension (default: wav)')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip files that already have features (default: True)')
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all audio files
    audio_files = sorted(audio_dir.rglob(f'*.{args.ext}'))
    print(f"Found {len(audio_files):,d} .{args.ext} files in {audio_dir}")

    if len(audio_files) == 0:
        print("No audio files found. Check --audio_dir.")
        return

    # Load model
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


if __name__ == '__main__':
    main()
