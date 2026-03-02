# Continuous Morphing with kNN-VC

This guide explains how to use kNN-VC to create speech that continuously morphs from one speaker to another over the duration of an utterance.

## What is Continuous Morphing?

Continuous morphing creates speech stimuli where the speaker identity gradually transitions from Speaker A to Speaker B over time. Instead of converting an entire utterance to a single target speaker, the conversion happens **progressively**:

- **Start (t=0)**: 100% Speaker A
- **Middle (t=0.5)**: 50% Speaker A, 50% Speaker B
- **End (t=1.0)**: 100% Speaker B

This is particularly useful for:
- Perceptual experiments studying speaker identity
- Creating smooth voice transitions in creative applications
- Investigating speaker representation in speech processing

## Why kNN-VC is Ideal for Morphing

kNN-VC is uniquely suited for continuous morphing because it operates **frame-by-frame** rather than using a global speaker embedding. Each frame's speaker identity is determined independently by which reference features it matches to, making gradual blending natural and straightforward.

## Quick Start

```python
import torch
import torchaudio

# Load model
knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True,
                        trust_repo=True, pretrained=True, device='cuda')

# Load source audio and reference speakers
query_seq = knn_vc.get_features('source.wav')
matching_set_A = knn_vc.get_matching_set(['speaker_A_ref1.wav', 'speaker_A_ref2.wav'])
matching_set_B = knn_vc.get_matching_set(['speaker_B_ref1.wav', 'speaker_B_ref2.wav'])

# Perform continuous morphing
out_wav = knn_vc.match_morph(
    query_seq,
    matching_set_A,
    matching_set_B,
    topk=4,
    morph_profile='linear'
)

# Save result
torchaudio.save('morphed_output.wav', out_wav[None], 16000)
```

## API Reference

### `match_morph()`

The main method for continuous morphing:

```python
match_morph(
    query_seq: Tensor,
    matching_set_A: Tensor,
    matching_set_B: Tensor,
    synth_set_A: Tensor = None,
    synth_set_B: Tensor = None,
    topk: int = 4,
    morph_profile: str = 'linear',
    morph_params: dict = None,
    silence_aware: bool = False,
    vad_threshold_db: float = -40,
    vad_min_silence_ms: int = 50,
    tgt_loudness_db: float | None = -16,
    target_duration: float | None = None,
    device: str | None = None
) -> Tensor
```

**Parameters:**

- **`query_seq`**: Source features from `get_features()` - shape (N, dim)
- **`matching_set_A`**: Reference features for Speaker A (start) - shape (N_A, dim)
- **`matching_set_B`**: Reference features for Speaker B (end) - shape (N_B, dim)
- **`synth_set_A/B`**: Optional synthesis features (for prematched vocoder compatibility)
- **`topk`**: Number of nearest neighbors to average (default: 4)
  - Lower values (2-3): More source characteristics preserved
  - Higher values (6-8): Smoother output, less distinct
- **`silence_aware`**: Enable silence-aware morphing (default: False)
  - When True, alpha only advances during speech, frozen during silence
  - See [Silence-Aware Morphing](#silence-aware-morphing) section below
- **`vad_threshold_db`**: Energy threshold in dB for voice activity detection (default: -40)
  - Lower values (e.g., -50): More sensitive, treat quieter regions as speech
  - Higher values (e.g., -30): Less sensitive, treat more regions as silence
- **`vad_min_silence_ms`**: Minimum silence duration in milliseconds (default: 50)
  - Silence spans shorter than this are treated as speech
  - Helps filter out brief dips in energy within continuous speech
- **`morph_profile`**: Type of interpolation curve
  - `'linear'`: Constant rate of change (default)
  - `'sigmoid'`: Smooth S-curve transition
  - `'step'`: Abrupt change at midpoint
  - `'custom'`: User-defined curve
- **`morph_params`**: Profile-specific parameters (dict)
  - For sigmoid: `{'steepness': 10}` (higher = sharper transition)
  - For step: `{'threshold': 0.5}` (where to transition)
  - For custom: `{'alpha_values': [0.0, 0.1, ..., 1.0]}`
- **`tgt_loudness_db`**: Output loudness normalization (default: -16 dB)
- **`target_duration`**: Optional target duration in seconds
- **`device`**: Compute device ('cuda', 'cpu', etc.)

**Returns:**
- Converted waveform tensor of shape (T,)

## Morphing Profiles

### Linear (Default)

Constant rate of change from A to B.

```python
out_wav = knn_vc.match_morph(query_seq, matching_A, matching_B,
                             morph_profile='linear')
```

**Use when:** You want uniform transition speed throughout the utterance.

### Sigmoid

Smooth S-curve with slower change at the start and end, faster in the middle.

```python
out_wav = knn_vc.match_morph(query_seq, matching_A, matching_B,
                             morph_profile='sigmoid',
                             morph_params={'steepness': 10})
```

**Parameters:**
- `steepness` (default: 10): Controls transition sharpness
  - Lower (5): Very gradual, barely noticeable in the middle
  - Higher (20): Sharp transition, almost step-like

**Use when:** You want perceptually natural transitions or want to focus the ambiguous region in the middle of the utterance.

### Step

Abrupt change at a specified threshold (control condition).

```python
out_wav = knn_vc.match_morph(query_seq, matching_A, matching_B,
                             morph_profile='step',
                             morph_params={'threshold': 0.5})
```

**Parameters:**
- `threshold` (default: 0.5): Where to switch from A to B (0.0 to 1.0)

**Use when:** You need a control condition with no gradual transition for experiments.

### Custom

User-defined interpolation curve.

```python
import numpy as np

# Example: Non-monotonic curve (A → B → A)
alpha_values = np.concatenate([
    np.linspace(0, 1, 100),  # A to B
    np.linspace(1, 0, 100)   # B back to A
])

out_wav = knn_vc.match_morph(query_seq, matching_A, matching_B,
                             morph_profile='custom',
                             morph_params={'alpha_values': alpha_values})
```

**Use when:** You need non-standard interpolation curves for specific experimental designs.

## Tips and Best Practices

### Choosing Reference Audio

1. **Quality matters**: Use clean, clear speech with minimal background noise
2. **Duration**: 1-5 minutes per speaker is ideal
   - Too little (< 30s): May not capture speaker variability
   - Too much (> 5 min): Diminishing returns, slower processing
3. **Variety**: Multiple shorter clips better than one long clip
4. **Consistency**: Same recording conditions for both speakers if possible

### Tuning Parameters

**k (topk) parameter:**
```python
# More source prosody preserved, potentially less smooth
out_wav = knn_vc.match_morph(..., topk=2)

# Balanced (default)
out_wav = knn_vc.match_morph(..., topk=4)

# Smoother, more averaged
out_wav = knn_vc.match_morph(..., topk=8)
```

**Morph profile selection:**
- Start with `'linear'` for simplicity
- Use `'sigmoid'` if linear feels too abrupt
- Adjust sigmoid `steepness` to control transition speed
- Use `'step'` as control condition in experiments

### Validating Results

1. **Listen carefully**: The transition should be smooth and natural
2. **Check boundaries**:
   - Start should sound like Speaker A
   - End should sound like Speaker B
3. **Plot interpolation curve**:
   ```python
   from matcher import generate_morph_profile
   import matplotlib.pyplot as plt

   alpha = generate_morph_profile(100, 'sigmoid', {'steepness': 10})
   plt.plot(alpha)
   plt.xlabel('Frame')
   plt.ylabel('α (Speaker B weight)')
   plt.title('Morphing Profile')
   plt.show()
   ```

4. **Use speaker verification**: Track speaker similarity scores over time

## Example Scripts

See the `examples/` directory:

1. **`simple_morph_example.py`**: Minimal example to get started quickly
2. **`continuous_morph_demo.py`**: Full-featured CLI tool with all options

Run the CLI tool:
```bash
python examples/continuous_morph_demo.py \
    --source input.wav \
    --speaker-a ref_a1.wav ref_a2.wav \
    --speaker-b ref_b1.wav ref_b2.wav \
    --output morphed.wav \
    --profile sigmoid \
    --topk 4
```

## How It Works

The continuous morphing implementation:

1. **Extracts features** from source and both reference speakers using WavLM
2. **Generates interpolation curve** α(t) from 0 to 1 over the utterance
3. **Performs double kNN matching**:
   - For each frame, finds k nearest neighbors in Speaker A's features
   - For each frame, finds k nearest neighbors in Speaker B's features
4. **Blends the matched features**: `out = (1-α) × matched_A + α × matched_B`
5. **Vocodes** the blended features into waveform with HiFiGAN

The key insight: because kNN-VC operates frame-by-frame, we can blend two speaker identities at each time step independently.

## Computational Cost

Continuous morphing requires:
- **kNN matching**: ~2× cost of standard conversion (matching against two speakers)
- **Feature blending**: Negligible (simple linear combination)
- **Vocoding**: Same cost (single pass)

**Total: ~2× the time of standard conversion**, which is still very fast since kNN is efficient.

## Silence-Aware Morphing

By default, the morphing profile advances uniformly over time, **including during silence periods**. This can cause perceptual discontinuities when speech resumes after a pause.

**Silence-aware morphing** addresses this by advancing the alpha coefficient only during active speech, freezing it during silence.

### The Problem

With standard time-based morphing:
```
[Speech A] [Pause] [Speech B] [Pause] [Speech C]
  α=0.0    α=0.33   α=0.67    α=0.84   α=1.0
           ↑ morphing continues during silence
```

During the pause at t=0.33, alpha has advanced to 33% even though no speech occurred. When Speech B starts, it's immediately at 67% morphing, creating an abrupt transition.

### The Solution

With silence-aware morphing:
```
[Speech A] [Pause] [Speech B] [Pause] [Speech C]
  α=0.0    α=0.0    α=0.5     α=0.5    α=1.0
           ↑ frozen           ↑ frozen
```

Alpha only advances during voiced segments, creating smoother, more natural transitions.

### Usage

```python
out_wav = knn_vc.match_morph(
    query_seq,
    matching_set_A,
    matching_set_B,
    topk=8,
    silence_aware=True,          # Enable silence-aware morphing
    vad_threshold_db=-40,        # Energy threshold for VAD
    vad_min_silence_ms=50,       # Ignore silence shorter than 50ms
    query_wav_path='source.wav'  # For accurate waveform-based VAD
)
```

### Parameters

- **`silence_aware`** (bool): Enable silence-aware morphing
  - `False` (default): Alpha advances uniformly over time
  - `True`: Alpha frozen during silence, advances only during speech

- **`vad_threshold_db`** (float): Energy threshold in dB relative to maximum (default: -40)
  - Lower values (e.g., -50): More sensitive, treat quieter regions as speech
  - Higher values (e.g., -30): Less sensitive, treat more regions as silence
  - Typical range: -50 to -30 dB

- **`vad_min_silence_ms`** (int): Minimum silence duration in milliseconds (default: 50)
  - Silence spans shorter than this are treated as speech
  - Helps filter out brief pauses between syllables
  - Typical range: 30 to 100 ms

### When to Use

**Use silence-aware morphing when:**
- Your utterance has significant pauses (>100ms)
- You want morphing to happen only during voiced segments
- Testing perceptual effects of morphing timing

**Use standard morphing when:**
- Continuous speech with no pauses
- You want morphing to complete by a specific time point
- Simpler, more predictable behavior desired

### Visualization

To visualize the difference between standard and silence-aware morphing:

```bash
python examples/visualize_silence_aware.py \
    --source sample_data/mies/test/mies_test_0.wav \
    --output silence_aware_comparison.png
```

This generates a plot showing:
1. Voice activity detection (speech vs. silence)
2. Standard alpha profile (advances uniformly)
3. Silence-aware alpha profile (frozen during silence)

### How It Works

Silence-aware morphing uses energy-based voice activity detection (VAD) with two modes:

**Waveform-based VAD (Recommended):**
1. **Compute RMS energy** per frame from raw audio waveform
2. **Threshold** energy to classify frames as speech or silence
3. **Filter** short silence spans (< `vad_min_silence_ms`)
4. **Generate alpha** that advances only during speech frames
5. **Ensure** alpha reaches 1.0 at the final speech frame

**Feature-based VAD (Fallback):**
- Uses variance of WavLM features instead of waveform energy
- Automatic if source audio path not provided
- Less accurate but still effective for many use cases

To use waveform-based VAD (recommended), provide the source audio path:
```python
out = knn_vc.match_morph(
    query_seq, match_a, match_b,
    silence_aware=True,
    query_wav_path='source.wav'  # ← Enables waveform-based VAD
)
```

### Example Comparison

```python
# Standard morphing
out_standard = knn_vc.match_morph(
    query_seq, match_A, match_B,
    silence_aware=False
)

# Silence-aware morphing
out_silence_aware = knn_vc.match_morph(
    query_seq, match_A, match_B,
    silence_aware=True,
    vad_threshold_db=-40
)
```

Listen to both outputs to compare the perceptual difference. Silence-aware morphing typically sounds more natural for utterances with pauses.

### Tuning the VAD Threshold

If the VAD is too sensitive or not sensitive enough, adjust `vad_threshold_db`:

```python
# More sensitive (detect more as speech)
out = knn_vc.match_morph(..., vad_threshold_db=-50)

# Less sensitive (detect more as silence)
out = knn_vc.match_morph(..., vad_threshold_db=-30)
```

Use the visualization script to inspect VAD decisions before generating the full morphed output.

## Troubleshooting

**Problem**: Output doesn't sound like either speaker
- **Solution**: Check that reference audio is clean and sufficient (1+ minutes per speaker)
- **Solution**: Try increasing k: `topk=8`

**Problem**: Transition is too abrupt
- **Solution**: Use sigmoid profile instead of linear
- **Solution**: Decrease sigmoid steepness: `morph_params={'steepness': 5}`
- **Solution**: Try silence-aware morphing to avoid discontinuities at pauses

**Problem**: Not enough morphing (sounds mostly like one speaker)
- **Solution**: Verify both matching sets are different speakers
- **Solution**: Check that source audio duration is sufficient for smooth transition

**Problem**: Silence-aware morphing sounds choppy
- **Solution**: VAD may be too sensitive, increase threshold: `vad_threshold_db=-30`
- **Solution**: Increase minimum silence duration: `vad_min_silence_ms=100`
- **Solution**: Use visualization script to inspect VAD decisions

**Problem**: Silence-aware morphing behaves like standard
- **Solution**: VAD may not be detecting silence, lower threshold: `vad_threshold_db=-50`
- **Solution**: Check if your audio actually has pauses (use visualization script)
- **Solution**: Verify audio is clean (background noise can be detected as speech)

**Problem**: Audio quality degradation
- **Solution**: This is expected behavior - morphing creates ambiguous speaker identity
- **Solution**: Use prematched vocoder (default)
- **Solution**: Ensure reference audio is high quality

## Citation

If you use continuous morphing with kNN-VC in your research, please cite both the original kNN-VC paper and acknowledge the morphing extension:

```bibtex
@inproceedings{baas2023knnvc,
  author={Matthew Baas and Benjamin van Niekerk and Herman Kamper},
  title={Voice Conversion With Just Nearest Neighbors},
  year=2023,
  booktitle={Interspeech},
}
```

## Further Reading

- Original kNN-VC paper: https://arxiv.org/abs/2305.18975
- Implementation details: See `CONTINUOUS_MORPHING_PLAN.md`
- WavLM paper: https://arxiv.org/abs/2110.13900
