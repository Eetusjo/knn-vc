import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ──────────────────────────────────────────────────────────────────────────────
# Projection modules: WavLM (B, T, 1024) → pseudo-mel (B, T, num_mels)
# ──────────────────────────────────────────────────────────────────────────────

class LinearProjection(nn.Module):
    """Single linear layer. Baseline — cheapest, fastest to train."""

    def __init__(self, in_dim: int = 1024, out_dim: int = 100):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class MLPProjection(nn.Module):
    """Two-layer MLP with ReLU. Adds capacity for nonlinear mapping."""

    def __init__(self, in_dim: int = 1024, out_dim: int = 100, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ConvProjection(nn.Module):
    """Pointwise reduction then temporal conv1d. Captures local temporal
    context (coarticulation) that a per-frame linear layer misses."""

    def __init__(self, in_dim: int = 1024, out_dim: int = 100,
                 hidden_dim: int = 256, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_dim, kernel_size, padding=pad),
        )

    def forward(self, x):
        # x: (B, T, 1024) → conv expects (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x.permute(0, 2, 1)


class ConvBNProjection(nn.Module):
    """Pointwise reduction, batch norm, then temporal conv1d. Batch norm
    stabilises training when the projection is the only trainable component
    (Phase A)."""

    def __init__(self, in_dim: int = 1024, out_dim: int = 100,
                 hidden_dim: int = 256, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_dim, kernel_size, padding=pad),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x.permute(0, 2, 1)


class DeconvProjection(nn.Module):
    """Transposed convolution that jointly projects and upsamples from 50 Hz
    to ~93.75 Hz (factor ~1.875). This replaces the separate F.interpolate
    step — the upsampling becomes learned rather than fixed linear
    interpolation.

    Uses stride=2 to approximately double the frame rate (50→100 Hz, close
    to BigVGAN's 93.75 Hz). The remaining mismatch is handled by a final
    F.interpolate to exact length in the vocoder forward pass.
    """

    UPSAMPLE_FACTOR = 2  # 50 Hz × 2 = 100 Hz ≈ 93.75 Hz

    def __init__(self, in_dim: int = 1024, out_dim: int = 100,
                 hidden_dim: int = 512, kernel_size: int = 4):
        super().__init__()
        self.pointwise = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        )
        self.deconv = nn.ConvTranspose1d(
            hidden_dim, out_dim,
            kernel_size=kernel_size,
            stride=self.UPSAMPLE_FACTOR,
            padding=(kernel_size - self.UPSAMPLE_FACTOR) // 2,
        )

    @property
    def upsample_factor(self):
        return self.UPSAMPLE_FACTOR

    def forward(self, x):
        x = self.pointwise(x)        # (B, T, hidden)
        x = x.permute(0, 2, 1)       # (B, hidden, T)
        x = self.deconv(x)           # (B, out_dim, T*2)
        return x.permute(0, 2, 1)    # (B, T*2, out_dim)


PROJECTION_TYPES = {
    'linear': LinearProjection,
    'mlp': MLPProjection,
    'conv': ConvProjection,
    'conv_bn': ConvBNProjection,
    'deconv': DeconvProjection,
}


def build_projection(proj_type: str, in_dim: int = 1024, out_dim: int = 100,
                      **kwargs) -> nn.Module:
    if proj_type not in PROJECTION_TYPES:
        raise ValueError(f"Unknown projection type '{proj_type}'. "
                         f"Choose from: {list(PROJECTION_TYPES.keys())}")
    return PROJECTION_TYPES[proj_type](in_dim=in_dim, out_dim=out_dim, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Vocoder wrapper
# ──────────────────────────────────────────────────────────────────────────────

class BigVGANVocoder(nn.Module):
    """Wrapper that adapts BigVGAN to accept WavLM features.

    BigVGAN expects mel-spectrogram input (batch, num_mels, time) at ~93.75Hz
    (24000Hz / 256 hop). WavLM produces features at 50Hz (20ms frames).

    This wrapper:
      1. Projects 1024-dim WavLM features to num_mels (100) via a learned projection
      2. Interpolates to BigVGAN's expected frame rate (~93.75Hz)
      3. Runs BigVGAN synthesis at 24kHz
      4. Resamples output to target_sr (default 16kHz) if needed
    """

    def __init__(self, bigvgan_model, projection: nn.Module, target_sr: int = 16000):
        super().__init__()
        self.bigvgan = bigvgan_model
        self.projection = projection
        self.bigvgan_sr = bigvgan_model.h.sampling_rate   # 24000
        self.target_sr = target_sr
        self.bigvgan_hop = bigvgan_model.h.hop_size       # 256
        self.wavlm_frame_rate = 50  # Hz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            - x: (batch, seq_len, 1024) WavLM features
        Returns:
            - (batch, 1, num_samples) waveform at target_sr
        """
        input_len = x.shape[1]

        # 1. Project WavLM features to pseudo-mel dimension
        x = self.projection(x)  # (batch, seq_len', num_mels) — seq_len' may differ for deconv

        # 2. Transpose to Conv1d format expected by BigVGAN
        x = x.permute(0, 2, 1)  # (batch, num_mels, seq_len')

        # 3. Interpolate to BigVGAN's expected frame rate
        #    target is always based on original input length, so deconv projections
        #    that already upsampled just get a small correction here.
        target_len = int(input_len * (self.bigvgan_sr / self.bigvgan_hop) / self.wavlm_frame_rate)
        if x.shape[2] != target_len:
            x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)

        # 4. BigVGAN synthesis at 24kHz
        wav = self.bigvgan(x)  # (batch, 1, num_samples_24k)

        # 5. Resample to target SR if needed
        if self.target_sr != self.bigvgan_sr:
            wav = torchaudio.functional.resample(
                wav.squeeze(1), self.bigvgan_sr, self.target_sr
            ).unsqueeze(1)

        return wav

    def remove_weight_norm(self):
        self.bigvgan.remove_weight_norm()
