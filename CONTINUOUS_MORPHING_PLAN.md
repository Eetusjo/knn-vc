# Continuous Morphing Implementation Plan for kNN-VC

## Executive Summary

This document outlines a concrete implementation plan for adapting kNN-VC to create continuously morphing speech stimuli, where voice conversion happens gradually across an utterance (e.g., starting as Speaker A and gradually morphing to Speaker B).

## Background: Why kNN-VC is Ideal for Continuous Morphing

### Current kNN-VC Architecture
The kNN-VC system has three components:
1. **WavLM Encoder**: Extracts self-supervised speech representations (layer 6 by default)
2. **kNN Matcher**: For each source frame, finds k nearest neighbors in target reference features and takes their mean
3. **HiFiGAN Vocoder**: Converts matched features back to waveform

### Key Architectural Advantage
Unlike most voice conversion systems that use a single global speaker embedding, **kNN-VC operates frame-by-frame**. Each frame's speaker identity is determined by which reference features it matches to. This frame-level control makes continuous morphing naturally feasible.

### Current Behavior (Standard Conversion)
```python
# For each frame t in query_seq:
dists = cosine_distance(query_seq[t], matching_set)  # All frames from target speaker
best_indices = topk_smallest(dists, k=4)
out_feats[t] = mean(matching_set[best_indices])
```

This assigns every frame to Speaker B (the matching set), resulting in complete conversion.

## Proposed Solution: Dual-Set Morphing

### Core Concept
Maintain **two separate matching sets** (Speaker A and Speaker B) and blend their contributions with a time-varying interpolation coefficient α(t):

```python
converted(t) = (1 - α(t)) × kNN_A(t) + α(t) × kNN_B(t)
```

where:
- α(t) = 0 → 100% Speaker A (start of utterance)
- α(t) = 1 → 100% Speaker B (end of utterance)
- α(t) ∈ (0,1) → blend of both speakers (middle of utterance)

### Why This Works
1. **Frame-level identity**: kNN-VC doesn't have a global embedding that needs to be smoothly interpolated; each frame can independently blend two speaker identities
2. **Feature-space interpolation**: WavLM features form a continuous space where linear interpolation is meaningful
3. **Existing infrastructure**: The matching logic (cosine distance, topk) can be reused; we just need to compute matches twice and blend

## Implementation Design

### 1. Core Changes to `matcher.py`

#### A. New Method: `match_morph()`
Create a new method in `KNeighborsVC` class that handles continuous morphing:

```python
@torch.inference_mode()
def match_morph(self,
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
                device: str | None = None) -> Tensor:
    """
    Perform continuous morphing from Speaker A to Speaker B.

    Arguments:
        - query_seq: Tensor (N, dim) - source features
        - matching_set_A: Tensor (N_A, dim) - Speaker A reference features
        - matching_set_B: Tensor (N_B, dim) - Speaker B reference features
        - synth_set_A: Optional separate synthesis features for A
        - synth_set_B: Optional separate synthesis features for B
        - topk: k for kNN matching
        - morph_profile: 'linear', 'sigmoid', 'custom', 'step'
        - morph_params: Additional parameters for morphing profile
        - tgt_loudness_db: Output normalization
        - target_duration: Optional duration interpolation
        - device: Compute device

    Returns:
        - converted waveform of shape (T,)
    """
```

**Why this design:**
- **Separate matching sets**: Allows using different reference speakers for start/end
- **Morph profiles**: Flexibility in interpolation curves (linear, sigmoid, etc.)
- **Reuses existing infrastructure**: Leverages `fast_cosine_dist()` and `vocode()` methods
- **Optional synth sets**: Maintains compatibility with prematched vocoder approach

#### B. Helper Function: `generate_morph_profile()`
Create interpolation coefficient generator:

```python
def generate_morph_profile(n_frames: int,
                          profile: str = 'linear',
                          params: dict = None) -> Tensor:
    """
    Generate time-varying interpolation coefficients α(t) for morphing.

    Arguments:
        - n_frames: Number of frames in query sequence
        - profile: Type of interpolation curve
            * 'linear': α(t) = t / T
            * 'sigmoid': α(t) = 1 / (1 + exp(-k(t - T/2)))
            * 'exponential': α(t) = (exp(k*t/T) - 1) / (exp(k) - 1)
            * 'step': α(t) = 0 if t < T/2 else 1
            * 'custom': user-provided array
        - params: Profile-specific parameters (e.g., steepness for sigmoid)

    Returns:
        - Tensor (n_frames,) with values in [0, 1]
    """
```

**Why different profiles:**
- **Linear**: Constant rate of change (default, simple)
- **Sigmoid**: Perceptually smooth transition with slow start/end
- **Exponential**: Useful for asymmetric morphing
- **Step**: Control condition for experiments (abrupt change)
- **Custom**: Maximum flexibility for researchers

#### C. Implementation Logic for `match_morph()`

```python
# 1. Setup (similar to original match())
device = torch.device(device) if device is not None else self.device
if synth_set_A is None: synth_set_A = matching_set_A.to(device)
if synth_set_B is None: synth_set_B = matching_set_B.to(device)
matching_set_A = matching_set_A.to(device)
matching_set_B = matching_set_B.to(device)
query_seq = query_seq.to(device)

# 2. Handle target duration interpolation
if target_duration is not None:
    target_samples = int(target_duration * self.sr)
    scale_factor = (target_samples/self.hop_length) / query_seq.shape[0]
    query_seq = F.interpolate(query_seq.T[None], scale_factor=scale_factor, mode='linear')[0].T

# 3. Generate morphing profile
n_frames = query_seq.shape[0]
alpha = generate_morph_profile(n_frames, morph_profile, morph_params).to(device)  # (n_frames,)

# 4. Perform kNN matching for BOTH speakers
dists_A = fast_cosine_dist(query_seq, matching_set_A, device=device)  # (n_frames, N_A)
dists_B = fast_cosine_dist(query_seq, matching_set_B, device=device)  # (n_frames, N_B)

best_A = dists_A.topk(k=topk, largest=False, dim=-1)  # indices: (n_frames, k)
best_B = dists_B.topk(k=topk, largest=False, dim=-1)  # indices: (n_frames, k)

out_feats_A = synth_set_A[best_A.indices].mean(dim=1)  # (n_frames, dim)
out_feats_B = synth_set_B[best_B.indices].mean(dim=1)  # (n_frames, dim)

# 5. BLEND features according to morphing profile
# alpha shape: (n_frames,) -> (n_frames, 1) for broadcasting
out_feats = (1 - alpha[:, None]) * out_feats_A + alpha[:, None] * out_feats_B

# 6. Vocode and normalize (identical to original match())
prediction = self.vocode(out_feats[None].to(device)).cpu().squeeze()

if tgt_loudness_db is not None:
    src_loudness = torchaudio.functional.loudness(prediction[None], self.h.sampling_rate)
    pred_wav = torchaudio.functional.gain(prediction, tgt_loudness_db - src_loudness)
else:
    pred_wav = prediction

return pred_wav
```

**Why this approach:**
- **Double kNN computation**: Computationally cheap (kNN is fast) and provides independent matches
- **Feature-space blending**: Interpolates in WavLM space before vocoding (single vocoder pass)
- **Broadcasting**: α is (n_frames,) broadcast to (n_frames, dim) for element-wise multiplication

### 2. Optional Enhancement: `get_matching_set_from_speaker_id()`

For convenience when working with speaker datasets:

```python
def get_matching_set_from_speaker_id(self,
                                     dataset_path: Path,
                                     speaker_id: str,
                                     max_duration: float = 300) -> Tensor:
    """
    Helper to load all audio for a given speaker and create matching set.
    Useful for experiments with structured speaker datasets.
    """
```

**Why:** Simplifies experiment scripting when working with corpora like LibriSpeech

### 3. Example Usage Script

Create `examples/continuous_morph_demo.py`:

```python
import torch
import torchaudio

# Load model
knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True,
                        trust_repo=True, pretrained=True, device='cuda')

# Load source audio
src_wav_path = 'path/to/source.wav'
query_seq = knn_vc.get_features(src_wav_path)

# Load reference audios for two speakers
ref_A_paths = ['speaker_A_ref1.wav', 'speaker_A_ref2.wav']
ref_B_paths = ['speaker_B_ref1.wav', 'speaker_B_ref2.wav']

matching_set_A = knn_vc.get_matching_set(ref_A_paths)
matching_set_B = knn_vc.get_matching_set(ref_B_paths)

# Perform continuous morphing
out_wav = knn_vc.match_morph(
    query_seq,
    matching_set_A,
    matching_set_B,
    topk=4,
    morph_profile='linear'  # or 'sigmoid', 'exponential', etc.
)

# Save output
torchaudio.save('morphed_output.wav', out_wav[None], 16000)
```

**Why this structure:**
- Mirrors existing API (`get_features`, `get_matching_set`, `match`)
- Clear separation between A and B references
- Easy to experiment with different profiles

### 4. Morphing Profile Implementations

```python
def generate_morph_profile(n_frames: int,
                          profile: str = 'linear',
                          params: dict = None) -> Tensor:
    if params is None:
        params = {}

    t = torch.linspace(0, 1, n_frames)  # normalized time [0, 1]

    if profile == 'linear':
        alpha = t

    elif profile == 'sigmoid':
        # Sigmoid centered at 0.5, steepness controlled by k
        k = params.get('steepness', 10)  # higher = steeper
        alpha = torch.sigmoid(k * (t - 0.5))
        # Normalize to [0, 1] range
        alpha = (alpha - alpha[0]) / (alpha[-1] - alpha[0])

    elif profile == 'exponential':
        k = params.get('rate', 2)  # exp rate
        alpha = (torch.exp(k * t) - 1) / (torch.exp(torch.tensor(k)) - 1)

    elif profile == 'step':
        threshold = params.get('threshold', 0.5)
        alpha = (t >= threshold).float()

    elif profile == 'custom':
        alpha = torch.tensor(params['alpha_values'], dtype=torch.float32)
        if alpha.shape[0] != n_frames:
            raise ValueError(f"Custom alpha must have {n_frames} values")

    else:
        raise ValueError(f"Unknown profile: {profile}")

    return alpha
```

**Why different profiles:**
- **Linear**: Simplest, perceptually constant rate
- **Sigmoid**: More natural perception (humans are sensitive to rate of change)
- **Exponential**: Start as mostly A, quickly transition to B (or vice versa with negative rate)
- **Step**: Control condition for perceptual experiments
- **Custom**: Arbitrary researcher-defined curves (e.g., non-monotonic for specific experiments)

## Testing Strategy

### 1. Unit Tests
Create `tests/test_continuous_morph.py`:

```python
def test_morph_profile_linear():
    alpha = generate_morph_profile(100, 'linear')
    assert alpha[0] == 0.0
    assert alpha[-1] == 1.0
    assert torch.allclose(alpha[50], torch.tensor(0.5), atol=0.02)

def test_morph_profile_shapes():
    for profile in ['linear', 'sigmoid', 'exponential', 'step']:
        alpha = generate_morph_profile(200, profile)
        assert alpha.shape == (200,)
        assert alpha.min() >= 0
        assert alpha.max() <= 1

def test_match_morph_output_shape():
    # Mock model and test that output shape matches expectations
    pass

def test_morph_extremes():
    # Test that α=0 → Speaker A only, α=1 → Speaker B only
    pass
```

**Why these tests:**
- Verify profile generation correctness
- Ensure boundary conditions (α ∈ [0,1])
- Validate that extreme values reproduce single-speaker conversion

### 2. Perceptual Validation

Create stimuli with different morph profiles and evaluate:
1. **Intelligibility**: WER using ASR (should remain low)
2. **Speaker identity trajectory**: Use speaker verification model to track similarity to A and B over time
3. **Smoothness**: Perceptual listening tests

**Why:** Continuous morphing should maintain speech quality while achieving smooth identity transition

### 3. Edge Cases to Test
- Very short utterances (< 0.5s)
- Very long utterances (> 30s)
- Different k values (topk=1, topk=10)
- Identical A and B (should match standard conversion)
- Using source speaker as A (self-to-B morphing)

## File Structure After Implementation

```
knn-vc/
├── matcher.py                          # Modified: add match_morph() and generate_morph_profile()
├── morph_utils.py                      # New: helper functions for morphing experiments
├── examples/
│   └── continuous_morph_demo.py        # New: demonstration script
├── tests/
│   └── test_continuous_morph.py        # New: unit tests
├── CONTINUOUS_MORPHING_PLAN.md         # This document
└── README_MORPHING.md                  # New: user-facing documentation
```

## Implementation Steps (Ordered)

1. **Add `generate_morph_profile()` to `matcher.py`** (~50 lines)
   - Implement all profile types
   - Add input validation

2. **Add `match_morph()` method to `KNeighborsVC` class** (~100 lines)
   - Follow structure outlined above
   - Reuse existing helper methods (`fast_cosine_dist`, `vocode`)

3. **Create example script** (`examples/continuous_morph_demo.py`)
   - Simple demonstration with linear morphing
   - Include visualization of α(t) profile

4. **Write unit tests** (`tests/test_continuous_morph.py`)
   - Test all morph profiles
   - Validate output shapes and ranges

5. **Create user documentation** (`README_MORPHING.md`)
   - Usage examples
   - Parameter descriptions
   - Troubleshooting tips

6. **Optional: Create utilities for batch processing** (`morph_utils.py`)
   - Functions for processing multiple stimuli
   - Speaker verification integration for validation

## Computational Considerations

### Overhead Compared to Standard Conversion
- **kNN matching**: 2× cost (need to match against both A and B)
- **Feature blending**: Negligible (simple linear combination)
- **Vocoding**: Same cost (single pass)

**Total overhead: ~2× of standard conversion** (still very fast since kNN is efficient)

### Memory Requirements
- Must hold two matching sets in memory simultaneously
- For 5 minutes of reference audio per speaker: ~2 × (5×60×50) × 1024 × 4 bytes ≈ 60 MB
- Negligible compared to model weights (WavLM: ~1.2GB, HiFiGAN: ~64MB)

## Alternative Approaches Considered (and Why Not)

### 1. **Global Speaker Embedding Interpolation**
- **Approach**: Compute speaker embeddings for A and B, interpolate them, condition vocoder
- **Why not**: kNN-VC doesn't use speaker embeddings; would require architectural changes

### 2. **Cascaded Partial Conversions**
- **Approach**: Divide utterance into segments, convert each segment with different α
- **Why not**: Would create discontinuities at segment boundaries

### 3. **Training a Morphing-Aware Vocoder**
- **Approach**: Retrain HiFiGAN to handle blended features
- **Why not**: Current vocoder already handles WavLM features; blending in feature space should work

### 4. **Weighted Distance Metrics**
- **Approach**: Modify cosine distance to favor A or B based on time
- **Why not**: Less interpretable; harder to control morphing curve

## Expected Outcomes

### Successful Implementation Should Produce:
1. **Smooth speaker identity transition**: Speaker verification scores should show monotonic shift from A to B
2. **Preserved intelligibility**: WER should remain similar to standard conversion
3. **Natural prosody**: Timing and intonation should follow source (as in standard kNN-VC)
4. **Flexible control**: Different morph profiles should produce perceptibly different trajectories

### Potential Challenges and Mitigations:
1. **Discontinuities at boundaries**: Use sigmoid profile instead of linear
2. **Prosody artifacts**: May need to adjust k (topk) for smoother matching
3. **Identity ambiguity in middle**: Expected and desirable for perceptual experiments
4. **HiFiGAN handling interpolated features**: If issues arise, could add small blend-aware fine-tuning

## Integration with Existing Code

### Backwards Compatibility
- All existing functionality remains unchanged
- New `match_morph()` method is additive, doesn't modify `match()`
- Existing scripts and notebooks continue to work

### API Design Principles
- Follows existing naming conventions (`match` → `match_morph`)
- Parameter names consistent with existing methods
- Default parameters provide sensible behavior

## Future Extensions

### 1. Multi-Speaker Morphing (N > 2)
```python
def match_morph_multi(query_seq, matching_sets: list[Tensor],
                     alpha_profiles: list[Tensor]) -> Tensor:
    """Morph through N speakers with custom time-varying weights."""
```

### 2. Prosody-Aware Morphing
- Morph at phone boundaries instead of uniformly
- Requires phone alignment (e.g., Montreal Forced Aligner)

### 3. Content-Aware Morphing
- Different morph rates for vowels vs. consonants
- Phoneme-specific α(t) curves

### 4. Bidirectional Morphing
- A → B → A (non-monotonic α)
- Useful for creating cyclic stimuli

## Summary

This implementation plan provides a **minimal, efficient, and flexible** approach to continuous morphing in kNN-VC by:

1. **Leveraging frame-level architecture**: kNN-VC's design naturally supports per-frame blending
2. **Reusing existing infrastructure**: No changes to WavLM, HiFiGAN, or core matching logic
3. **Adding ~150 lines of code**: Single new method + helper function
4. **Maintaining compatibility**: All existing code continues to work
5. **Providing flexibility**: Multiple morph profiles for different experimental needs

The approach is **theoretically sound** (blending in feature space), **computationally efficient** (~2× overhead), and **easy to validate** (boundary cases should match standard conversion).

## References

- kNN-VC paper: Baas, van Niekerk, Kamper. "Voice Conversion With Just Nearest Neighbors." Interspeech 2023.
- WavLM paper: Chen et al. "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing." IEEE/ACM TASLP 2022.
- HiFiGAN paper: Kong et al. "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis." NeurIPS 2020.
