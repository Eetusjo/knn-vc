"""
Diagnostic script to analyze VAD energy distribution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from hubconf import knn_vc as load_knn_vc

# Load model
print("Loading model...")
knn_vc = load_knn_vc(pretrained=True, prematched=True, device='cpu')

# Load test file
test_file = 'sample_data/mies/test/mies_test_0.wav'
print(f"Loading: {test_file}")
query_seq = knn_vc.get_features(test_file)

# Analyze energy
print("\nAnalyzing energy distribution...")

# Current method: L2 norm of features
energy_l2 = query_seq.norm(dim=1)
energy_l2_db = 20 * torch.log10(energy_l2 + 1e-8)
energy_l2_db_norm = energy_l2_db - energy_l2_db.max()

print(f"L2 norm energy:")
print(f"  Range: {energy_l2_db_norm.min():.2f} to {energy_l2_db_norm.max():.2f} dB")
print(f"  Mean: {energy_l2_db_norm.mean():.2f} dB")
print(f"  Std: {energy_l2_db_norm.std():.2f} dB")
print(f"  10th percentile: {torch.quantile(energy_l2_db_norm, 0.1):.2f} dB")
print(f"  90th percentile: {torch.quantile(energy_l2_db_norm, 0.9):.2f} dB")

# Alternative: Variance across feature dimensions
energy_var = query_seq.var(dim=1)
energy_var_db = 10 * torch.log10(energy_var + 1e-8)
energy_var_db_norm = energy_var_db - energy_var_db.max()

print(f"\nVariance energy:")
print(f"  Range: {energy_var_db_norm.min():.2f} to {energy_var_db_norm.max():.2f} dB")
print(f"  Mean: {energy_var_db_norm.mean():.2f} dB")
print(f"  Std: {energy_var_db_norm.std():.2f} dB")
print(f"  10th percentile: {torch.quantile(energy_var_db_norm, 0.1):.2f} dB")
print(f"  90th percentile: {torch.quantile(energy_var_db_norm, 0.9):.2f} dB")

# Alternative: Mean absolute value
energy_mean = query_seq.abs().mean(dim=1)
energy_mean_db = 20 * torch.log10(energy_mean + 1e-8)
energy_mean_db_norm = energy_mean_db - energy_mean_db.max()

print(f"\nMean absolute energy:")
print(f"  Range: {energy_mean_db_norm.min():.2f} to {energy_mean_db_norm.max():.2f} dB")
print(f"  Mean: {energy_mean_db_norm.mean():.2f} dB")
print(f"  Std: {energy_mean_db_norm.std():.2f} dB")
print(f"  10th percentile: {torch.quantile(energy_mean_db_norm, 0.1):.2f} dB")
print(f"  90th percentile: {torch.quantile(energy_mean_db_norm, 0.9):.2f} dB")

# Load actual waveform for comparison
waveform, sr = torchaudio.load(test_file)
waveform = waveform.squeeze()

# Compute waveform energy (frame-level)
hop_length = 320
frame_energy = []
for i in range(query_seq.shape[0]):
    start_sample = i * hop_length
    end_sample = start_sample + hop_length
    frame_wav = waveform[start_sample:end_sample]
    frame_energy.append((frame_wav ** 2).mean().item())

frame_energy = torch.tensor(frame_energy)
frame_energy_db = 10 * torch.log10(frame_energy + 1e-8)
frame_energy_db_norm = frame_energy_db - frame_energy_db.max()

print(f"\nWaveform RMS energy:")
print(f"  Range: {frame_energy_db_norm.min():.2f} to {frame_energy_db_norm.max():.2f} dB")
print(f"  Mean: {frame_energy_db_norm.mean():.2f} dB")
print(f"  Std: {frame_energy_db_norm.std():.2f} dB")
print(f"  10th percentile: {torch.quantile(frame_energy_db_norm, 0.1):.2f} dB")
print(f"  90th percentile: {torch.quantile(frame_energy_db_norm, 0.9):.2f} dB")

# Plot comparison
fig, axes = plt.subplots(5, 1, figsize=(14, 12))
time_sec = np.arange(len(query_seq)) * 0.02

axes[0].plot(time_sec, energy_l2_db_norm.numpy(), linewidth=0.8)
axes[0].set_ylabel('Energy (dB)')
axes[0].set_title('L2 Norm Energy (CURRENT METHOD)')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=-40, color='r', linestyle='--', label='Threshold -40dB')
axes[0].legend()

axes[1].plot(time_sec, energy_var_db_norm.numpy(), linewidth=0.8)
axes[1].set_ylabel('Energy (dB)')
axes[1].set_title('Variance Energy')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=-40, color='r', linestyle='--', label='Threshold -40dB')
axes[1].legend()

axes[2].plot(time_sec, energy_mean_db_norm.numpy(), linewidth=0.8)
axes[2].set_ylabel('Energy (dB)')
axes[2].set_title('Mean Absolute Energy')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=-40, color='r', linestyle='--', label='Threshold -40dB')
axes[2].legend()

axes[3].plot(time_sec, frame_energy_db_norm.numpy(), linewidth=0.8)
axes[3].set_ylabel('Energy (dB)')
axes[3].set_title('Waveform RMS Energy (GROUND TRUTH)')
axes[3].grid(True, alpha=0.3)
axes[3].axhline(y=-40, color='r', linestyle='--', label='Threshold -40dB')
axes[3].legend()

# Plot histogram
axes[4].hist(energy_l2_db_norm.numpy(), bins=50, alpha=0.5, label='L2 norm', density=True)
axes[4].hist(energy_var_db_norm.numpy(), bins=50, alpha=0.5, label='Variance', density=True)
axes[4].hist(frame_energy_db_norm.numpy(), bins=50, alpha=0.5, label='Waveform RMS', density=True)
axes[4].set_xlabel('Energy (dB)')
axes[4].set_ylabel('Density')
axes[4].set_title('Energy Distribution Comparison')
axes[4].legend()
axes[4].grid(True, alpha=0.3)
axes[4].axvline(x=-40, color='r', linestyle='--', label='Threshold -40dB')

plt.tight_layout()
plt.savefig('vad_energy_analysis.png', dpi=150)
print(f"\nSaved visualization to: vad_energy_analysis.png")
