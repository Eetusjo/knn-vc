"""
Visualization Script for Silence-Aware Morphing

This script visualizes the difference between standard time-based morphing
and silence-aware morphing. It shows:
1. Voice activity detection (speech vs. silence)
2. Standard alpha profile (advances uniformly over time)
3. Silence-aware alpha profile (advances only during speech)

Usage:
    python examples/visualize_silence_aware.py \
        --source sample_data/mies/test/mies_test_0.wav \
        --output silence_aware_comparison.png
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import local knn-vc modules
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from hubconf import knn_vc as load_knn_vc
from matcher import (
    detect_voice_activity_energy,
    generate_morph_profile,
    generate_morph_profile_silence_aware
)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize silence-aware morphing profiles'
    )

    parser.add_argument(
        '--source',
        type=str,
        default='sample_data/mies/test/mies_test_0.wav',
        help='Path to source audio file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='silence_aware_comparison.png',
        help='Output path for visualization'
    )
    parser.add_argument(
        '--vad-threshold',
        type=float,
        default=-40,
        help='VAD energy threshold in dB (default: -40)'
    )
    parser.add_argument(
        '--morph-profile',
        type=str,
        default='linear',
        choices=['linear', 'sigmoid', 'step'],
        help='Morph profile type (default: linear)'
    )

    args = parser.parse_args()

    # Verify source file exists
    if not Path(args.source).exists():
        print(f"Error: Source file not found: {args.source}")
        return

    print("=" * 60)
    print("Silence-Aware Morphing Visualization")
    print("=" * 60)

    # Load kNN-VC model (for feature extraction)
    print("\n[1/4] Loading kNN-VC model...")
    knn_vc = load_knn_vc(pretrained=True, prematched=True, device='cpu')

    # Extract features
    print(f"\n[2/4] Extracting features from: {args.source}")
    query_seq = knn_vc.get_features(args.source)
    n_frames = query_seq.shape[0]
    duration_sec = n_frames * 0.02  # 20ms per frame
    print(f"      Extracted {n_frames} frames ({duration_sec:.2f} seconds)")

    # Detect voice activity
    print(f"\n[3/4] Detecting voice activity (threshold: {args.vad_threshold} dB)")

    # Load waveform for accurate VAD
    waveform, sr = torchaudio.load(args.source)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    waveform = waveform.squeeze()

    is_speech = detect_voice_activity_energy(
        query_seq,
        threshold_db=args.vad_threshold,
        min_duration_ms=50,
        waveform=waveform,  # Use waveform for accurate VAD
        hop_length=320
    )
    n_speech = is_speech.sum().item()
    n_silence = n_frames - n_speech
    pct_speech = 100 * n_speech / n_frames
    print(f"      Speech: {n_speech} frames ({pct_speech:.1f}%)")
    print(f"      Silence: {n_silence} frames ({100-pct_speech:.1f}%)")

    # Generate both morph profiles
    print(f"\n[4/4] Generating morph profiles ({args.morph_profile})")
    alpha_standard = generate_morph_profile(n_frames, args.morph_profile)
    alpha_silence_aware = generate_morph_profile_silence_aware(
        n_frames,
        is_speech,
        args.morph_profile
    )

    # Create visualization
    print(f"\n[5/5] Creating visualization...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))

    # Time axis in seconds
    time_sec = np.arange(n_frames) * 0.02

    # Plot 1: Waveform (already loaded above for VAD)
    waveform_np = waveform.numpy()
    time_wav = np.arange(len(waveform_np)) / 16000

    axes[0].plot(time_wav, waveform_np, linewidth=0.5, color='steelblue')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'Source Audio: {Path(args.source).name}')
    axes[0].set_xlim(0, duration_sec)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Voice Activity Detection
    axes[1].fill_between(
        time_sec,
        0,
        is_speech.float().numpy(),
        alpha=0.4,
        color='green',
        label='Speech'
    )
    axes[1].fill_between(
        time_sec,
        0,
        (~is_speech).float().numpy(),
        alpha=0.4,
        color='red',
        label='Silence'
    )
    axes[1].set_ylabel('Voice Activity')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_xlim(0, duration_sec)
    axes[1].legend(loc='upper right')
    axes[1].set_title(f'Voice Activity Detection (Threshold: {args.vad_threshold} dB)')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Standard Alpha Profile
    axes[2].plot(time_sec, alpha_standard.numpy(), linewidth=2, color='blue', label='Standard')
    axes[2].fill_between(
        time_sec,
        0,
        is_speech.float().numpy() * alpha_standard.numpy().max(),
        alpha=0.15,
        color='gray',
        label='Speech regions'
    )
    axes[2].set_ylabel('Alpha (Speaker B weight)')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_xlim(0, duration_sec)
    axes[2].legend(loc='upper left')
    axes[2].set_title(f'Standard Morphing Profile ({args.morph_profile}): α advances over time')
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Silence-Aware Alpha Profile
    axes[3].plot(time_sec, alpha_silence_aware.numpy(), linewidth=2, color='green', label='Silence-aware')
    axes[3].fill_between(
        time_sec,
        0,
        is_speech.float().numpy() * alpha_silence_aware.numpy().max(),
        alpha=0.15,
        color='gray',
        label='Speech regions'
    )
    axes[3].set_ylabel('Alpha (Speaker B weight)')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].set_xlim(0, duration_sec)
    axes[3].legend(loc='upper left')
    axes[3].set_title(f'Silence-Aware Morphing Profile ({args.morph_profile}): α frozen during silence')
    axes[3].grid(True, alpha=0.3)

    # Add annotations for key differences
    # Find first significant silence region
    silence_regions = []
    in_silence = False
    silence_start = 0

    for i in range(n_frames):
        if not is_speech[i] and not in_silence:
            # Start of silence
            silence_start = i
            in_silence = True
        elif is_speech[i] and in_silence:
            # End of silence
            if i - silence_start > 10:  # Only consider silences > 200ms
                silence_regions.append((silence_start, i))
            in_silence = False

    if silence_regions:
        # Annotate first significant silence region
        s_start, s_end = silence_regions[0]
        s_mid = (s_start + s_end) // 2
        t_mid = s_mid * 0.02

        # Show difference in alpha values during silence
        alpha_std_val = alpha_standard[s_mid].item()
        alpha_sa_val = alpha_silence_aware[s_mid].item()

        axes[2].annotate(
            f'α = {alpha_std_val:.2f}',
            xy=(t_mid, alpha_std_val),
            xytext=(t_mid + 0.5, alpha_std_val + 0.15),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
            fontsize=10,
            color='blue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue', alpha=0.8)
        )

        axes[3].annotate(
            f'α = {alpha_sa_val:.2f}\n(frozen)',
            xy=(t_mid, alpha_sa_val),
            xytext=(t_mid + 0.5, alpha_sa_val - 0.2),
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            fontsize=10,
            color='green',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.8)
        )

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"      Saved: {args.output}")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)
    print(f"\nOutput file: {args.output}")
    print(f"\nKey observations:")
    print(f"  1. Standard morphing advances uniformly regardless of speech/silence")
    print(f"  2. Silence-aware morphing freezes during silence regions")
    print(f"  3. Both profiles start at α=0 and end at α=1.0")
    print(f"  4. Silence-aware ensures morphing happens only during audible speech")


if __name__ == '__main__':
    main()
