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

## Troubleshooting

**Problem**: Output doesn't sound like either speaker
- **Solution**: Check that reference audio is clean and sufficient (1+ minutes per speaker)
- **Solution**: Try increasing k: `topk=8`

**Problem**: Transition is too abrupt
- **Solution**: Use sigmoid profile instead of linear
- **Solution**: Decrease sigmoid steepness: `morph_params={'steepness': 5}`

**Problem**: Not enough morphing (sounds mostly like one speaker)
- **Solution**: Verify both matching sets are different speakers
- **Solution**: Check that source audio duration is sufficient for smooth transition

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
