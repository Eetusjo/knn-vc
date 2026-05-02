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


# ──────────────────────────────────────────────────────────────────────────────
# Direct vocoder: WavLM features → BigVGAN with no projection or interpolation
# ──────────────────────────────────────────────────────────────────────────────

class BigVGANDirectVocoder(nn.Module):
    """Wrapper that feeds WavLM features straight into BigVGAN.

    Requires a BigVGAN configured with `num_mels=1024` (conv_pre input dim) and
    `hop_size = sampling_rate // wavlm_frame_rate` so that one WavLM frame
    produces exactly hop_size audio samples. No projection, no F.interpolate.

    Compared to BigVGANVocoder this removes the 1024→100 bottleneck and the
    50→93.75 Hz frame-rate adapter; the conv_pre layer of BigVGAN itself is the
    implicit input adapter, trained jointly with the rest of the generator.
    """

    def __init__(self, bigvgan_model, target_sr: int | None = None):
        super().__init__()
        self.bigvgan = bigvgan_model
        self.bigvgan_sr = bigvgan_model.h.sampling_rate
        self.target_sr = target_sr if target_sr is not None else self.bigvgan_sr

        # Sanity: hop_size must equal sampling_rate / wavlm_frame_rate (50 Hz).
        # Otherwise the number of audio samples produced per WavLM frame won't
        # match what the upsample stack assumes, and lengths will drift.
        expected_hop = self.bigvgan_sr // 50
        if bigvgan_model.h.hop_size != expected_hop:
            raise ValueError(
                f"BigVGANDirectVocoder requires hop_size == sampling_rate // 50 "
                f"({expected_hop}) for WavLM's 50 Hz frame rate, got "
                f"hop_size={bigvgan_model.h.hop_size} at sr={self.bigvgan_sr}.")

        if bigvgan_model.h.num_mels != 1024:
            raise ValueError(
                f"BigVGANDirectVocoder requires num_mels == 1024 (the WavLM "
                f"feature dim) so conv_pre accepts WavLM features directly, "
                f"got num_mels={bigvgan_model.h.num_mels}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            - x: (batch, seq_len, 1024) WavLM features at 50 Hz
        Returns:
            - (batch, 1, num_samples) waveform at target_sr
        """
        # BigVGAN expects (B, C, T)
        x = x.permute(0, 2, 1)              # (B, 1024, seq_len)
        wav = self.bigvgan(x)               # (B, 1, seq_len * hop_size)

        if self.target_sr != self.bigvgan_sr:
            wav = torchaudio.functional.resample(
                wav.squeeze(1), self.bigvgan_sr, self.target_sr
            ).unsqueeze(1)
        return wav

    def remove_weight_norm(self):
        self.bigvgan.remove_weight_norm()


# ──────────────────────────────────────────────────────────────────────────────
# Partial state-dict loading (for warm-starting from pretrained mel-input BigVGAN)
# ──────────────────────────────────────────────────────────────────────────────

def load_bigvgan_partial(model: nn.Module, state_dict: dict, verbose: bool = True):
    """Load weights from `state_dict` into `model`, skipping any keys whose
    shapes don't match. Returns (loaded_keys, skipped_keys).

    Used to warm-start a direct-path BigVGAN (1024-d input, hop=480, modified
    upsample stack) from NVIDIA's pretrained 100-band 24 kHz checkpoint:
    conv_pre and the upsamplers whose strides changed are re-initialized; the
    AMP resblocks (channel-count compatible since upsample_initial_channel and
    the number of stages are unchanged) and conv_post transfer cleanly.
    """
    own_state = model.state_dict()

    # First pass: identify shape mismatches. Group weight_v/weight_g pairs from
    # weight_norm so that skipping `<layer>.weight_v` also skips `<layer>.weight_g`
    # — otherwise the layer ends up with pretrained magnitude on a random
    # direction, which is a confusing initialization.
    direct_skip = set()
    for name, target in own_state.items():
        if name in state_dict and state_dict[name].shape != target.shape:
            direct_skip.add(name)

    pair_skip = set()
    for name in direct_skip:
        if name.endswith('.weight_v'):
            pair_skip.add(name[:-len('.weight_v')] + '.weight_g')
        elif name.endswith('.weight_g'):
            pair_skip.add(name[:-len('.weight_g')] + '.weight_v')

    skip_set = direct_skip | (pair_skip & own_state.keys() & state_dict.keys())

    loaded = []
    skipped = []  # entries: (name, ckpt_shape, model_shape, reason)
    missing_in_ckpt = []

    for name, target in own_state.items():
        if name not in state_dict:
            missing_in_ckpt.append(name)
            continue
        src = state_dict[name]
        if name in direct_skip:
            skipped.append((name, tuple(src.shape), tuple(target.shape), 'shape'))
            continue
        if name in skip_set:
            skipped.append((name, tuple(src.shape), tuple(target.shape), 'paired'))
            continue
        target.copy_(src)
        loaded.append(name)

    extra = [k for k in state_dict.keys() if k not in own_state]

    if verbose:
        n_shape = sum(1 for s in skipped if s[3] == 'shape')
        n_paired = sum(1 for s in skipped if s[3] == 'paired')
        print(f"[BigVGAN partial-load] loaded {len(loaded):,d} tensors, "
              f"skipped {n_shape:,d} (shape mismatch) + {n_paired:,d} (paired weight_norm), "
              f"missing {len(missing_in_ckpt):,d} (random init), "
              f"unused {len(extra):,d}.")
        if skipped:
            print("  Skipped (will be trained from random init):")
            for n, s_src, s_tgt, reason in skipped[:8]:
                tag = 'shape mismatch' if reason == 'shape' else 'paired with mismatched weight_v/g'
                print(f"    {n}: ckpt {s_src} → model {s_tgt} ({tag})")
            if len(skipped) > 8:
                print(f"    … and {len(skipped) - 8} more")

    return loaded, [s[0] for s in skipped]
