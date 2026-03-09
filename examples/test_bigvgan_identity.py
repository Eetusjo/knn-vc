"""
BigVGAN2 Vocoder Identity Test

Tests BigVGAN2 vocoding quality with ground-truth input features:
  audio -> mel spectrogram -> BigVGAN2 -> audio

This is the "perfect condition" baseline — the vocoder receives exactly the
mel features it was trained on, with no upstream model in the loop.

BigVGAN2 model: nvidia/bigvgan_v2_24khz_100band_256x
  sampling_rate : 24000 Hz
  n_fft / win   : 1024
  hop_size      : 256
  num_mels      : 100
  fmin / fmax   : 0 / None (full bandwidth)
"""

import sys
import argparse
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torchaudio
import torchaudio.transforms as T
import bigvgan


BIGVGAN_SR   = 24000
N_FFT        = 1024
HOP_SIZE     = 256
WIN_SIZE     = 1024
NUM_MELS     = 100
FMIN         = 0
FMAX         = None   # full bandwidth, matches model config


def get_mel(wav: torch.Tensor, sr: int, device: torch.device) -> torch.Tensor:
    """Compute mel spectrogram matching BigVGAN2 training config.

    Args:
        wav: (1, T) waveform tensor, any sample rate
        sr:  sample rate of wav

    Returns:
        (1, num_mels, frames) mel spectrogram
    """
    if sr != BIGVGAN_SR:
        wav = torchaudio.functional.resample(wav, sr, BIGVGAN_SR)

    mel_transform = T.MelSpectrogram(
        sample_rate=BIGVGAN_SR,
        n_fft=N_FFT,
        win_length=WIN_SIZE,
        hop_length=HOP_SIZE,
        f_min=FMIN,
        f_max=FMAX,
        n_mels=NUM_MELS,
        power=1.0,          # amplitude spectrogram (not power)
        norm='slaney',
        mel_scale='slaney',
    ).to(device)

    wav = wav.to(device)
    mel = mel_transform(wav)                      # (1, num_mels, frames)
    mel = torch.log(torch.clamp(mel, min=1e-5))   # log-amplitude, same as BigVGAN training
    return mel


def main():
    parser = argparse.ArgumentParser(description="BigVGAN2 identity vocoder test (mel -> audio)")
    parser.add_argument("input", type=str, help="Path to input audio file")
    parser.add_argument("--output", type=str, default="bigvgan_identity_out.wav",
                        help="Output WAV path (saved at 24kHz)")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading BigVGAN2 (nvidia/bigvgan_v2_24khz_100band_256x)...")
    model = bigvgan.BigVGAN.from_pretrained(
        'nvidia/bigvgan_v2_24khz_100band_256x',
        use_cuda_kernel=False,
    )
    model = model.to(device).eval()
    model.remove_weight_norm()
    print(f"  Loaded with {sum(p.numel() for p in model.parameters()):,d} parameters.")

    print(f"\nLoading audio: {args.input}")
    wav, sr = torchaudio.load(args.input, normalize=True)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mix to mono
    print(f"  {wav.shape[-1] / sr:.2f}s at {sr}Hz")

    print("Computing mel spectrogram...")
    mel = get_mel(wav, sr, device)
    print(f"  Mel shape: {mel.shape}  (1, num_mels={mel.shape[1]}, frames={mel.shape[2]})")

    print("Vocoding...")
    with torch.inference_mode():
        wav_out = model(mel).squeeze(1)   # (1, T) at 24kHz

    # Loudness normalization to -16 dB LUFS
    loudness = torchaudio.functional.loudness(wav_out.cpu(), BIGVGAN_SR)
    wav_out = torchaudio.functional.gain(wav_out.cpu(), -16.0 - loudness)

    torchaudio.save(args.output, wav_out, BIGVGAN_SR)
    duration = wav_out.shape[-1] / BIGVGAN_SR
    print(f"\nSaved {duration:.2f}s audio to: {args.output}  (24kHz)")


if __name__ == "__main__":
    main()
