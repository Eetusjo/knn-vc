"""
Vocoder Identity Test

Tests the upper bound quality of kNN-VC by skipping kNN matching entirely.
Instead of replacing features with nearest neighbors from a reference speaker,
we feed the source features directly into the vocoder.

This answers: "How good can kNN-VC sound in the perfect matching condition?"
(i.e., if every kNN lookup returned the exact source feature)
"""

import sys
import argparse
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torchaudio
from hubconf import knn_vc as load_knn_vc


def main():
    parser = argparse.ArgumentParser(description="Test vocoder quality under perfect matching condition")
    parser.add_argument("input", type=str, help="Path to input audio file (16kHz mono WAV recommended)")
    parser.add_argument("--output", type=str, default="vocoder_identity_out.wav", help="Path to output WAV file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--prematched", action="store_true", default=True, help="Use prematched HiFiGAN weights (default: True)")
    parser.add_argument("--no-prematched", dest="prematched", action="store_false", help="Use standard HiFiGAN weights")
    args = parser.parse_args()

    print(f"Loading kNN-VC models on '{args.device}'...")
    knn_vc = load_knn_vc(prematched=args.prematched, pretrained=True, device=args.device)

    print(f"Extracting WavLM features from: {args.input}")
    # get_features returns (seq_len, dim) — same features that would flow through kNN matching
    features = knn_vc.get_features(args.input)
    print(f"  Feature shape: {features.shape}  (seq_len={features.shape[0]}, dim={features.shape[1]})")

    print("Vocoding features directly (identity / perfect-match condition)...")
    # vocode expects (bs, seq_len, dim), returns (bs, T) -> squeeze to (T,)
    with torch.inference_mode():
        wav_out = knn_vc.vocode(features[None].to(knn_vc.device)).cpu().squeeze()

    # Loudness normalization to -16 dB LUFS (same default as knn_vc.match)
    src_loudness = torchaudio.functional.loudness(wav_out[None], knn_vc.sr)
    wav_out = torchaudio.functional.gain(wav_out, -16.0 - src_loudness)

    torchaudio.save(args.output, wav_out[None], knn_vc.sr)
    duration = wav_out.shape[-1] / knn_vc.sr
    print(f"Saved {duration:.2f}s audio to: {args.output}")


if __name__ == "__main__":
    main()
