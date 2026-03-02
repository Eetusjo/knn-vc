"""
Test Script for Silence-Aware Morphing

This script tests the silence-aware morphing feature by generating two outputs:
1. Standard morphing (alpha advances uniformly over time)
2. Silence-aware morphing (alpha advances only during speech)

Listen to both outputs to compare the perceptual difference.

Usage:
    python examples/test_silence_aware_morph.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import local knn-vc modules
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torchaudio
from hubconf import knn_vc as load_knn_vc


def main():
    print("=" * 70)
    print("Silence-Aware Morphing Test")
    print("=" * 70)

    # Configuration
    test_id = 0
    ref_minutes = 5
    source = "mies"
    target = "nainen"
    longsrc = False  # Use 10s test samples

    src_wav_path = f'sample_data/{source}/test/{source}_test{"_20s" if longsrc else ""}_{test_id}.wav'
    ref_A_paths = [f'sample_data/{source}/ref/{source}_ref_{i}.wav' for i in range(ref_minutes)]
    ref_B_paths = [f'sample_data/{target}/ref/{target}_ref_{i}.wav' for i in range(ref_minutes)]

    # Verify files exist
    if not Path(src_wav_path).exists():
        print(f"Error: Source file not found: {src_wav_path}")
        return

    print(f"\nConfiguration:")
    print(f"  Source: {src_wav_path}")
    print(f"  Speaker A (start): {source} ({ref_minutes} min reference)")
    print(f"  Speaker B (end):   {target} ({ref_minutes} min reference)")

    # Load model
    print("\n[1/5] Loading kNN-VC model...")
    knn_vc = load_knn_vc(prematched=True, pretrained=True, device='cpu')

    # Extract features
    print(f"\n[2/5] Extracting features...")
    query_seq = knn_vc.get_features(src_wav_path)
    n_frames = query_seq.shape[0]
    duration = n_frames * 0.02
    print(f"      Source: {n_frames} frames ({duration:.2f} seconds)")

    matching_set_A = knn_vc.get_matching_set(ref_A_paths)
    print(f"      Speaker A reference: {matching_set_A.shape[0]} frames")

    matching_set_B = knn_vc.get_matching_set(ref_B_paths)
    print(f"      Speaker B reference: {matching_set_B.shape[0]} frames")

    # Test 1: Standard morphing
    print(f"\n[3/5] Performing standard morphing...")
    out_standard = knn_vc.match_morph(
        query_seq,
        matching_set_A,
        matching_set_B,
        topk=8,
        morph_profile='linear',
        silence_aware=False  # Standard time-based morphing
    )
    print(f"      Generated {len(out_standard) / 16000:.2f}s of audio")

    # Test 2: Silence-aware morphing
    print(f"\n[4/5] Performing silence-aware morphing...")
    out_silence_aware = knn_vc.match_morph(
        query_seq,
        matching_set_A,
        matching_set_B,
        topk=8,
        morph_profile='linear',
        silence_aware=True,         # Enable silence-aware morphing
        vad_threshold_db=-40,       # Energy threshold for VAD
        vad_min_silence_ms=50,      # Minimum silence duration
        query_wav_path=src_wav_path  # Provide waveform for accurate VAD
    )
    print(f"      Generated {len(out_silence_aware) / 16000:.2f}s of audio")

    # Save outputs
    print(f"\n[5/5] Saving outputs...")
    output_dir = Path('morphed_outputs')
    output_dir.mkdir(exist_ok=True)

    std_path = output_dir / f'morph_standard_{source}_to_{target}_test{test_id}.wav'
    sa_path = output_dir / f'morph_silence_aware_{source}_to_{target}_test{test_id}.wav'

    torchaudio.save(str(std_path), out_standard[None], 16000)
    print(f"      Standard:       {std_path}")

    torchaudio.save(str(sa_path), out_silence_aware[None], 16000)
    print(f"      Silence-aware:  {sa_path}")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
    print(f"\nListen to both outputs to compare:")
    print(f"  1. {std_path}")
    print(f"     → Alpha advances uniformly (morphing continues during silence)")
    print(f"\n  2. {sa_path}")
    print(f"     → Alpha frozen during silence (morphing only during speech)")
    print(f"\nExpected difference:")
    print(f"  - Standard: May have perceptual discontinuities when speech resumes")
    print(f"  - Silence-aware: More natural, smooth transitions at speech boundaries")
    print("=" * 70)

    # Also test with sigmoid profile
    print("\n" + "=" * 70)
    print("Bonus: Testing with sigmoid profile")
    print("=" * 70)

    print(f"\n[Bonus] Generating sigmoid profiles...")
    out_std_sigmoid = knn_vc.match_morph(
        query_seq,
        matching_set_A,
        matching_set_B,
        topk=8,
        morph_profile='sigmoid',
        morph_params={'steepness': 10},
        silence_aware=False
    )

    out_sa_sigmoid = knn_vc.match_morph(
        query_seq,
        matching_set_A,
        matching_set_B,
        topk=8,
        morph_profile='sigmoid',
        morph_params={'steepness': 10},
        silence_aware=True,
        vad_threshold_db=-40,
        query_wav_path=src_wav_path
    )

    std_sig_path = output_dir / f'morph_standard_sigmoid_{source}_to_{target}_test{test_id}.wav'
    sa_sig_path = output_dir / f'morph_silence_aware_sigmoid_{source}_to_{target}_test{test_id}.wav'

    torchaudio.save(str(std_sig_path), out_std_sigmoid[None], 16000)
    torchaudio.save(str(sa_sig_path), out_sa_sigmoid[None], 16000)

    print(f"      Standard (sigmoid):       {std_sig_path}")
    print(f"      Silence-aware (sigmoid):  {sa_sig_path}")

    print("\n" + "=" * 70)
    print("All tests complete! Generated 4 output files.")
    print("=" * 70)


if __name__ == '__main__':
    main()
