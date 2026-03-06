"""
Continuous Morphing Demo for kNN-VC

This script demonstrates how to use kNN-VC to create speech that continuously
morphs from one speaker to another over the duration of an utterance.

Usage:
    python continuous_morph_demo.py \
        --source source.wav \
        --speaker-a speaker_a_ref1.wav speaker_a_ref2.wav \
        --speaker-b speaker_b_ref1.wav speaker_b_ref2.wav \
        --output morphed_output.wav \
        --profile linear \
        --topk 4

Requirements:
    - torch, torchaudio, numpy (see README.md)
    - Input audio files should be 16kHz, single-channel WAV files
"""

import argparse
import torch
import torchaudio
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Create continuously morphing speech with kNN-VC'
    )

    # Required arguments
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to source audio file (16kHz WAV)'
    )
    parser.add_argument(
        '--speaker-a',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to reference audio for Speaker A (start speaker)'
    )
    parser.add_argument(
        '--speaker-b',
        type=str,
        nargs='+',
        required=True,
        help='Path(s) to reference audio for Speaker B (end speaker)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for morphed audio'
    )

    # Optional arguments
    parser.add_argument(
        '--profile',
        type=str,
        default='linear',
        choices=['linear', 'sigmoid', 'step'],
        help='Morphing profile type (default: linear)'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=4,
        help='Number of nearest neighbors for kNN matching (default: 4)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu, default: cuda)'
    )
    parser.add_argument(
        '--prematched',
        action='store_true',
        default=True,
        help='Use prematched vocoder (default: True)'
    )
    parser.add_argument(
        '--steepness',
        type=float,
        default=10.0,
        help='Steepness parameter for sigmoid profile (default: 10)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Source file not found: {args.source}")
    for path in args.speaker_a + args.speaker_b:
        if not Path(path).exists():
            raise FileNotFoundError(f"Reference file not found: {path}")

    print("=" * 60)
    print("kNN-VC Continuous Morphing Demo")
    print("=" * 60)

    # Load kNN-VC model
    print("\n[1/5] Loading kNN-VC model...")
    print(f"      Device: {args.device}")
    print(f"      Prematched vocoder: {args.prematched}")

    knn_vc = torch.hub.load(
        'bshall/knn-vc',
        'knn_vc',
        prematched=args.prematched,
        trust_repo=True,
        pretrained=True,
        device=args.device
    )

    # Extract source features
    print(f"\n[2/5] Extracting features from source audio...")
    print(f"      Source: {args.source}")

    query_seq = knn_vc.get_features(args.source)
    n_frames = query_seq.shape[0]
    duration = n_frames * 0.02  # ~20ms per frame
    print(f"      Extracted {n_frames} frames ({duration:.2f} seconds)")

    # Build matching sets for both speakers
    print(f"\n[3/5] Building matching sets...")
    print(f"      Speaker A references: {len(args.speaker_a)} file(s)")
    for i, path in enumerate(args.speaker_a, 1):
        print(f"        {i}. {Path(path).name}")

    matching_set_A = knn_vc.get_matching_set(args.speaker_a)
    print(f"      Speaker A: {matching_set_A.shape[0]} reference frames")

    print(f"      Speaker B references: {len(args.speaker_b)} file(s)")
    for i, path in enumerate(args.speaker_b, 1):
        print(f"        {i}. {Path(path).name}")

    matching_set_B = knn_vc.get_matching_set(args.speaker_b)
    print(f"      Speaker B: {matching_set_B.shape[0]} reference frames")

    # Perform continuous morphing
    print(f"\n[4/5] Performing continuous morphing...")
    print(f"      Profile: {args.profile}")
    print(f"      k (topk): {args.topk}")

    # Setup morphing parameters
    morph_params = {}
    if args.profile == 'sigmoid':
        morph_params['steepness'] = args.steepness
        print(f"      Sigmoid steepness: {args.steepness}")

    print(f"      Morphing trajectory: Speaker A (0%) → Speaker B (100%)")

    out_wav = knn_vc.match_morph(
        query_seq,
        matching_set_A,
        matching_set_B,
        topk=args.topk,
        morph_profile=args.profile,
        morph_params=morph_params
    )

    # Save output
    print(f"\n[5/5] Saving output...")
    print(f"      Output: {args.output}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torchaudio.save(
        str(output_path),
        out_wav[None],  # Add batch dimension
        16000  # Sample rate
    )

    print(f"      Saved: {len(out_wav) / 16000:.2f} seconds of audio")

    print("\n" + "=" * 60)
    print("Morphing complete!")
    print("=" * 60)
    print(f"\nOutput file: {args.output}")
    print(f"\nTips:")
    print(f"  - Try different profiles: --profile sigmoid")
    print(f"  - Adjust k for smoothness: --topk 8")
    print(f"  - For sharper sigmoid: --steepness 20")
    print(f"  - Use more reference audio for better quality")


if __name__ == '__main__':
    main()
