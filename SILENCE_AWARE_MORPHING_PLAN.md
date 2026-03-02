# Silence-Aware Morphing: Implementation Plan

## Problem Statement

**Current Issue:** When using linear (or any) interpolation for continuous morphing, the alpha coefficient changes uniformly over time, **including during silence periods**. This means:

```
Example with 5s utterance, linear morph:
[Speech A] [Silence] [Speech B] [Silence] [Speech C]
  α=0.0      α=0.33     α=0.67     α=0.84     α=1.0
```

During silence at t=0.33, the morph is 33% complete even though no speech has occurred. This creates **unintended discontinuities** when speech resumes:
- The silence before "Speech B" is morphed 33% toward Speaker B
- But "Speech B" immediately jumps to 67% morphing
- Perceptually jarring transition

**Desired Behavior:** Alpha should only advance during **active speech**, remaining constant during silence:

```
Example with silence-aware morphing:
[Speech A] [Silence] [Speech B] [Silence] [Speech C]
  α=0.0      α=0.0      α=0.5      α=0.5      α=1.0
            ↑ frozen    ↑ continues  ↑ frozen   ↑ continues
```

This ensures smooth, natural morphing where the transition happens only during audible speech.

---

## User's Proposed Approach

1. **Detect silence/speech spans** in the source utterance
2. **Generate the morph profile** using existing methods (linear, sigmoid, etc.)
3. **Freeze alpha during silence** - keep it constant at the value from the last speech frame

---

## Analysis: Strengths and Potential Issues

### ✅ Strengths

1. **Conceptually sound:** Morphing only during speech is perceptually more natural
2. **Backwards compatible:** Can be optional parameter (default: disabled)
3. **Leverages existing infrastructure:** kNN-VC already uses VAD in `get_features()`
4. **Flexible:** Works with all morph profiles (linear, sigmoid, custom)
5. **Addresses real problem:** Prevents alpha "wasting" on silence

### ⚠️ Potential Issues and Edge Cases

#### Issue 1: VAD Granularity vs. Feature Frame Rate

**Problem:**
- WavLM features are extracted at **50Hz** (20ms frames)
- Most VAD systems operate at different temporal resolutions:
  - Silero VAD: 30-100ms chunks
  - torchaudio VAD: Operates on entire waveform, returns trimmed audio (not frame labels)
  - Energy-based VAD: Can be frame-level but noisy

**Impact:** Misalignment between VAD decisions and feature frames

**Mitigation:**
- Use frame-level energy/RMS as simple VAD
- Or: Use Silero VAD and interpolate to match feature frame rate
- Ensure VAD outputs boolean mask of shape `(n_frames,)` aligned with WavLM features

---

#### Issue 2: Definition of "Silence"

**Ambiguity:** What counts as silence?
- Zero amplitude? (Almost never happens in real recordings)
- Below energy threshold? (What threshold? Varies by recording)
- Pauses between words? (How long? 50ms? 200ms?)
- Background noise during pauses? (Still has energy but not speech)

**Implications:**
- Too sensitive (high threshold) → treats quiet phonemes as silence → choppy morphing
- Too lenient (low threshold) → treats pauses as speech → alpha advances during pauses

**Examples of ambiguous cases:**
```
"Hello... [pause] ...world"   ← Is the pause silence?
"ssss" (fricative consonant)  ← Low energy, but speech
[Background noise]            ← Has energy, but not speech
```

**Recommendation:**
- Use **energy-based threshold** with sensible default (e.g., -40dB)
- Make threshold configurable by user
- Consider minimum duration (e.g., silence spans < 50ms are ignored)

---

#### Issue 3: Alpha Progression - Absolute vs. Relative Time

**Ambiguity:** How should alpha advance during speech?

**Option A: Relative to speech duration only**
```
Total utterance: 10s (5s speech + 5s silence)
Alpha at 50% of SPEECH time (2.5s of speech elapsed)
→ α = 0.5
```

**Option B: Relative to total duration, frozen during silence**
```
Total utterance: 10s (5s speech + 5s silence)
If silence is uniformly distributed:
→ Alpha advances half as fast during speech frames
→ At t=10s, α may only reach 0.5 if 50% was silence
```

**Recommended: Option A** - Alpha from 0→1 based on speech content only
- More predictable: Always starts at Speaker A, ends at Speaker B
- User expectation: "Morph completely from A to B during this utterance"
- Easier to reason about

---

#### Issue 4: Boundary Discontinuities at Silence→Speech Transitions

**Problem:** When speech resumes after silence, the last voiced frame before silence and first voiced frame after silence will have the **same alpha value**, but:
- They may be separated by seconds of real time
- They may have very different phonetic content
- kNN matching may find very different reference frames

**Example:**
```
Frame 100: "...oh" (end of word) → α = 0.4
[Silence frames 101-120]        → α = 0.4 (frozen)
Frame 121: "wo..." (start of word) → α = 0.4 (resumes)
```

Frames 100 and 121 both use α=0.4, but they're different phonemes → potential for discontinuity

**Mitigation:**
- This may actually be **less** of a problem than current approach (where alpha changes during silence)
- The temporal smoothing (from artifact fix) will help
- Could add small "fade-in" after silence (alpha advances slightly slower for first 2-3 frames)

**Recommendation:** Start simple, add fade-in only if testing reveals issues

---

#### Issue 5: Very Short Utterances or Continuous Speech

**Edge case 1: No silence detected**
```
Utterance: "hello" (continuous speech, no pauses)
→ VAD marks all frames as speech
→ Behavior identical to current implementation ✓
```

**Edge case 2: All silence**
```
Utterance: [background noise only]
→ VAD marks all frames as silence
→ Alpha never advances, stays at 0.0
→ Output is 100% Speaker A
```

**Edge case 3: Silence at start/end**
```
[Silence] [Speech] [Silence]
→ Alpha frozen at 0.0 during initial silence
→ Advances from 0→1 during speech only
→ Frozen at 1.0 during final silence ✓
```

**Recommendation:** Document these behaviors, they seem reasonable

---

#### Issue 6: Interaction with Sigmoid/Step Profiles

**Sigmoid profile:**
```python
# Current: sigmoid centered at t=0.5 (temporal midpoint)
alpha = sigmoid(k * (t - 0.5))

# With silence-aware: sigmoid centered at 50% of SPEECH frames
# Need to compute t based on cumulative speech time, not wall time
```

**Impact:** The "center" of the sigmoid should be at 50% of speech content, not 50% of total time

**Recommendation:** Recompute `t` as cumulative speech ratio:
```python
t_speech = cumsum(is_speech) / sum(is_speech)  # 0 to 1 based on speech progress
alpha_base = sigmoid(k * (t_speech - 0.5))     # Apply sigmoid to speech time
alpha = where(is_speech, alpha_base, alpha_from_last_speech_frame)
```

---

## Detailed Technical Design

### Component 1: Voice Activity Detection

**Function signature:**
```python
def detect_voice_activity(
    features: Tensor,           # (n_frames, feat_dim) WavLM features
    waveform: Tensor = None,    # Optional (n_samples,) raw audio
    method: str = 'energy',     # 'energy', 'silero', or 'external'
    threshold_db: float = -40,  # Energy threshold in dB
    min_duration_ms: int = 50,  # Minimum silence duration to consider
    frame_rate_hz: int = 50,    # Feature frame rate (20ms frames)
) -> Tensor:
    """
    Detect voice activity at the frame level.

    Returns:
        - Boolean tensor of shape (n_frames,) where True = speech, False = silence
    """
```

**Method 1: Energy-based (Simple, Fast) - RECOMMENDED FOR V1**

```python
def detect_voice_activity_energy(features, threshold_db=-40, min_duration_ms=50, frame_rate_hz=50):
    """Simple energy-based VAD on WavLM features"""
    # Compute energy per frame (L2 norm of feature vector)
    energy = features.norm(dim=1)  # (n_frames,)

    # Convert to dB
    energy_db = 20 * torch.log10(energy + 1e-8)

    # Normalize to max = 0 dB
    energy_db = energy_db - energy_db.max()

    # Threshold
    is_speech = energy_db > threshold_db

    # Filter short silence spans (< min_duration_ms)
    min_frames = int(min_duration_ms / 1000 * frame_rate_hz)
    is_speech = filter_short_silences(is_speech, min_frames)

    return is_speech

def filter_short_silences(is_speech, min_frames):
    """Remove silence spans shorter than min_frames"""
    # Find silence spans
    diff = torch.diff(is_speech.int(), prepend=torch.tensor([0]), append=torch.tensor([0]))
    silence_starts = torch.where(diff == -1)[0]  # Speech → silence
    silence_ends = torch.where(diff == 1)[0]     # Silence → speech

    # For each silence span, if too short, mark as speech
    for start, end in zip(silence_starts, silence_ends):
        if end - start < min_frames:
            is_speech[start:end] = True

    return is_speech
```

**Pros:**
- Fast: No additional model loading
- Operates directly on WavLM features (already computed)
- Perfect frame alignment
- No external dependencies

**Cons:**
- Less accurate than neural VAD
- May be fooled by background noise
- Threshold tuning required

---

**Method 2: Silero VAD (Accurate, More Complex) - OPTIONAL FOR V2**

```python
def detect_voice_activity_silero(waveform, frame_rate_hz=50):
    """Use Silero VAD for accurate neural-based detection"""
    import torch

    # Load Silero VAD model (cache after first load)
    if not hasattr(detect_voice_activity_silero, 'model'):
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        detect_voice_activity_silero.model = model

    model = detect_voice_activity_silero.model

    # Silero expects 16kHz mono waveform
    # Returns timestamps of speech segments
    speech_timestamps = model(waveform, sampling_rate=16000)

    # Convert timestamps to frame-level boolean mask
    n_frames = int(len(waveform) / (16000 / frame_rate_hz))
    is_speech = torch.zeros(n_frames, dtype=torch.bool)

    for segment in speech_timestamps:
        start_frame = int(segment['start'] / 16000 * frame_rate_hz)
        end_frame = int(segment['end'] / 16000 * frame_rate_hz)
        is_speech[start_frame:end_frame] = True

    return is_speech
```

**Pros:**
- Highly accurate (state-of-the-art neural VAD)
- Robust to noise
- Well-maintained library

**Cons:**
- Requires torch.hub download (adds dependency)
- Slower than energy-based
- Operates on waveform, not features (misalignment risk)

---

**Recommendation: Start with energy-based VAD**
- Simpler implementation
- Faster
- Sufficient for most cases
- Can add Silero as optional upgrade later

---

### Component 2: Silence-Aware Alpha Generation

**Function signature:**
```python
def generate_morph_profile_silence_aware(
    n_frames: int,
    is_speech: Tensor,          # (n_frames,) boolean mask
    profile: str = 'linear',
    params: dict = None
) -> Tensor:
    """
    Generate morph profile that only advances during speech frames.

    Arguments:
        - n_frames: Total number of frames
        - is_speech: Boolean tensor (n_frames,) where True = speech, False = silence
        - profile: 'linear', 'sigmoid', 'step', or 'custom'
        - params: Profile-specific parameters

    Returns:
        - Tensor of shape (n_frames,) with alpha values
          - Alpha advances from 0→1 during speech frames only
          - Alpha frozen during silence frames at last speech value
    """
```

**Algorithm:**

```python
def generate_morph_profile_silence_aware(n_frames, is_speech, profile='linear', params=None):
    if params is None:
        params = {}

    # Step 1: Compute cumulative speech time (0 to 1 based on speech content)
    # This gives us a "speech-only timeline"
    speech_cumsum = torch.cumsum(is_speech.float(), dim=0)
    total_speech_frames = is_speech.sum().item()

    if total_speech_frames == 0:
        # Edge case: No speech detected → all silence → freeze at 0
        return torch.zeros(n_frames)

    # Normalize to [0, 1] based on how much speech has occurred
    t_speech = speech_cumsum / total_speech_frames  # (n_frames,)

    # Step 2: Generate base alpha profile using speech timeline
    if profile == 'linear':
        alpha_base = t_speech

    elif profile == 'sigmoid':
        k = params.get('steepness', 10)
        alpha_base = torch.sigmoid(k * (t_speech - 0.5))
        # Normalize
        alpha_base = (alpha_base - alpha_base.min()) / (alpha_base.max() - alpha_base.min() + 1e-8)

    elif profile == 'step':
        threshold = params.get('threshold', 0.5)
        alpha_base = (t_speech >= threshold).float()

    elif profile == 'custom':
        # For custom, user must provide alpha values aligned with SPEECH frames only
        alpha_values = params.get('alpha_values')
        if alpha_values is None:
            raise ValueError("Custom profile requires 'alpha_values'")

        # Interpolate custom values to all frames based on speech progress
        alpha_custom = torch.tensor(alpha_values, dtype=torch.float32)
        # Map t_speech (0 to 1) to indices in custom array
        indices = (t_speech * (len(alpha_custom) - 1)).long()
        alpha_base = alpha_custom[indices]

    else:
        raise ValueError(f"Unknown profile: {profile}")

    # Step 3: Apply speech mask - freeze during silence
    # For silence frames, use the alpha from the previous speech frame
    alpha = torch.zeros(n_frames)
    last_speech_alpha = 0.0

    for i in range(n_frames):
        if is_speech[i]:
            alpha[i] = alpha_base[i]
            last_speech_alpha = alpha_base[i]
        else:
            # Silence: freeze at last speech value
            alpha[i] = last_speech_alpha

    return alpha
```

**Key design decisions:**

1. **Timeline is speech-relative:** `t_speech` goes from 0→1 based on cumulative speech, not wall time
2. **Sigmoid center is at 50% of speech content:** More intuitive than 50% of total time
3. **Silence frames inherit last speech alpha:** Smooth, no discontinuities
4. **Edge cases handled:** No speech → α=0 everywhere; all speech → same as current implementation

---

### Component 3: Integration with match_morph()

**Modified function signature:**
```python
def match_morph(
    self,
    query_seq: Tensor,
    matching_set_A: Tensor,
    matching_set_B: Tensor,
    synth_set_A: Tensor = None,
    synth_set_B: Tensor = None,
    topk: int = 4,
    morph_profile: str = 'linear',
    morph_params: dict = None,

    # NEW PARAMETERS:
    silence_aware: bool = False,           # Enable silence-aware morphing
    vad_method: str = 'energy',            # 'energy' or 'silero'
    vad_threshold_db: float = -40,         # Energy threshold for VAD
    vad_min_silence_ms: int = 50,          # Minimum silence duration
    query_waveform: Tensor = None,         # Optional: raw audio for Silero VAD

    tgt_loudness_db: float | None = -16,
    target_duration: float | None = None,
    device: str | None = None
) -> Tensor:
```

**Implementation changes in match_morph():**

```python
# After line 294 (before generating morph profile)

if silence_aware:
    # Detect voice activity at frame level
    if vad_method == 'energy':
        is_speech = detect_voice_activity_energy(
            query_seq,
            threshold_db=vad_threshold_db,
            min_duration_ms=vad_min_silence_ms
        )
    elif vad_method == 'silero':
        if query_waveform is None:
            raise ValueError("Silero VAD requires query_waveform parameter")
        is_speech = detect_voice_activity_silero(query_waveform)
        # Ensure alignment with features
        is_speech = is_speech[:query_seq.shape[0]]
    else:
        raise ValueError(f"Unknown VAD method: {vad_method}")

    # Generate silence-aware morph profile
    alpha = generate_morph_profile_silence_aware(
        n_frames,
        is_speech,
        morph_profile,
        morph_params
    ).to(device)
else:
    # Original behavior: time-based morphing
    alpha = generate_morph_profile(
        n_frames,
        morph_profile,
        morph_params
    ).to(device)

# Rest of the function continues unchanged...
```

---

## Complete Implementation Plan

### Phase 1: Core Implementation (Priority 1)

**Step 1.1: Implement Energy-based VAD** (30 min)
- Add `detect_voice_activity_energy()` function to `matcher.py`
- Add `filter_short_silences()` helper
- Unit tests:
  - Test on synthetic signals (speech + silence)
  - Verify frame alignment
  - Test edge cases (all silence, all speech, no silence)

**Step 1.2: Implement Silence-Aware Profile Generation** (45 min)
- Add `generate_morph_profile_silence_aware()` to `matcher.py`
- Support all existing profiles (linear, sigmoid, step, custom)
- Unit tests:
  - Verify alpha freezes during silence
  - Verify alpha advances only during speech
  - Test with different speech/silence patterns
  - Test edge cases (no speech, no silence)

**Step 1.3: Integrate with match_morph()** (30 min)
- Add parameters: `silence_aware`, `vad_method`, `vad_threshold_db`, `vad_min_silence_ms`
- Add conditional logic to use silence-aware or standard profile generation
- Ensure backward compatibility (default: `silence_aware=False`)

**Step 1.4: Update get_features() to Optionally Return Waveform** (15 min)
- For Silero VAD support (Phase 2), we need the raw waveform
- Add optional return of waveform from `get_features()`
- Or add `get_features_with_waveform()` variant

---

### Phase 2: Testing and Validation (Priority 2)

**Step 2.1: Create Test Script** (30 min)

```python
# examples/test_silence_aware_morph.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hubconf import knn_vc
import torchaudio
import matplotlib.pyplot as plt

# Load model
model = knn_vc(pretrained=True, prematched=True, device='cuda')

# Load test sample with pauses
query_seq = model.get_features('sample_data/mies/test/mies_test_0.wav')
match_a = model.get_matching_set([f'sample_data/mies/ref/mies_ref_{i}.wav' for i in range(5)])
match_b = model.get_matching_set([f'sample_data/nainen/ref/nainen_ref_{i}.wav' for i in range(5)])

# Test 1: Standard morphing (baseline)
out_standard = model.match_morph(
    query_seq, match_a, match_b,
    topk=8,
    morph_profile='linear',
    silence_aware=False
)

# Test 2: Silence-aware morphing
out_silence_aware = model.match_morph(
    query_seq, match_a, match_b,
    topk=8,
    morph_profile='linear',
    silence_aware=True,
    vad_threshold_db=-40
)

# Save outputs
torchaudio.save('morph_standard.wav', out_standard[None], 16000)
torchaudio.save('morph_silence_aware.wav', out_silence_aware[None], 16000)

print("Test complete. Listen to both outputs and compare.")
```

**Step 2.2: Visualize VAD and Alpha Profiles** (30 min)

```python
# examples/visualize_silence_aware.py

# Extract features and VAD
query_seq = model.get_features('test.wav')
from matcher import detect_voice_activity_energy, generate_morph_profile, generate_morph_profile_silence_aware

is_speech = detect_voice_activity_energy(query_seq, threshold_db=-40)

# Generate both profiles
alpha_standard = generate_morph_profile(len(query_seq), 'linear')
alpha_silence = generate_morph_profile_silence_aware(len(query_seq), is_speech, 'linear')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 8))

# VAD
axes[0].fill_between(range(len(is_speech)), 0, is_speech.float(), alpha=0.3, label='Speech')
axes[0].set_ylabel('Voice Activity')
axes[0].set_ylim(-0.1, 1.1)
axes[0].legend()

# Standard alpha
axes[1].plot(alpha_standard, label='Standard (time-based)', color='blue')
axes[1].set_ylabel('Alpha (Standard)')
axes[1].set_ylim(-0.1, 1.1)
axes[1].legend()

# Silence-aware alpha
axes[2].plot(alpha_silence, label='Silence-aware (speech-based)', color='green')
axes[2].fill_between(range(len(is_speech)), 0, is_speech.float() * alpha_silence.max(),
                      alpha=0.2, color='gray', label='Speech regions')
axes[2].set_ylabel('Alpha (Silence-aware)')
axes[2].set_xlabel('Frame')
axes[2].set_ylim(-0.1, 1.1)
axes[2].legend()

plt.tight_layout()
plt.savefig('silence_aware_comparison.png', dpi=150)
```

**Step 2.3: Perceptual Evaluation** (1-2 hours)
- Test on various samples:
  - Short pauses between words
  - Long pauses between sentences
  - Continuous speech (no pauses)
  - Mostly silence with brief speech
- Listen for:
  - Smoothness of transitions
  - Naturalness during silence
  - Any discontinuities at silence↔speech boundaries

---

### Phase 3: Documentation and Examples (Priority 3)

**Step 3.1: Update README_MORPHING.md** (30 min)

Add section:

```markdown
## Silence-Aware Morphing

By default, the morphing profile advances uniformly over time, including during
silence. This can cause perceptual discontinuities when speech resumes after a pause.

**Silence-aware morphing** advances the morph profile only during active speech,
freezing the alpha coefficient during silence periods.

### Usage

```python
out_wav = knn_vc.match_morph(
    query_seq,
    matching_A,
    matching_B,
    topk=8,
    silence_aware=True,          # Enable silence-aware morphing
    vad_threshold_db=-40,        # Energy threshold for silence detection
    vad_min_silence_ms=50        # Ignore silence shorter than 50ms
)
```

### Parameters

- **silence_aware** (bool): Enable silence-aware morphing (default: False)
- **vad_method** (str): Voice activity detection method ('energy' or 'silero')
- **vad_threshold_db** (float): Energy threshold in dB, relative to max (default: -40)
  - Lower (e.g., -50): More sensitive, treats quieter regions as speech
  - Higher (e.g., -30): Less sensitive, treats more regions as silence
- **vad_min_silence_ms** (int): Minimum silence duration to consider (default: 50ms)
  - Filters out very short pauses

### When to Use

**Use silence-aware morphing when:**
- Utterance has significant pauses (>100ms)
- You want morphing to happen only during voiced segments
- Testing perceptual effects of morphing location

**Use standard morphing when:**
- Continuous speech with no pauses
- You want morphing to complete by a specific time point
- Simpler, more predictable behavior

### Example

```python
# Standard: morphing advances during silence
[Word A α=0.3] [PAUSE α=0.5] [Word B α=0.7]
                      ↑ morphing continues during pause

# Silence-aware: morphing frozen during silence
[Word A α=0.3] [PAUSE α=0.3] [Word B α=0.7]
                      ↑ frozen at last speech value
```
```

**Step 3.2: Add Example Script** (20 min)
- Create `examples/silence_aware_demo.py`
- CLI tool with --silence-aware flag
- Visualization of VAD and alpha profile

**Step 3.3: Update Unit Tests** (30 min)
- Add tests to `tests/test_continuous_morph.py`:
  - Test `detect_voice_activity_energy()`
  - Test `generate_morph_profile_silence_aware()`
  - Test integration in `match_morph()`
  - Edge cases (all silence, no silence, etc.)

---

### Phase 4: Optional Enhancements (Priority 4 - Future Work)

**Enhancement 4.1: Silero VAD Integration** (1 hour)
- Add `detect_voice_activity_silero()` function
- Add torch.hub download with caching
- Handle waveform→feature alignment
- Document usage and comparison with energy-based

**Enhancement 4.2: VAD Visualization Tool** (30 min)
- CLI tool to visualize VAD output on a file
- Helps users tune `vad_threshold_db` parameter
- Overlay waveform, energy, and VAD decision

**Enhancement 4.3: Adaptive Threshold** (1 hour)
- Auto-compute threshold based on signal statistics
- Use percentile-based thresholding (e.g., 10th percentile)
- Reduces need for manual tuning

---

## Edge Cases and Error Handling

### Edge Case Matrix

| Scenario | Current Behavior | Silence-Aware Behavior | Handled? |
|----------|------------------|------------------------|----------|
| No silence (continuous speech) | α: 0→1 linearly | α: 0→1 linearly (same) | ✓ Yes |
| All silence | α: 0→1 linearly | α: 0 everywhere | ✓ Yes |
| Silence at start | α advances from start | α frozen at 0 until speech | ✓ Yes |
| Silence at end | α advances to 1.0 | α frozen at last speech value | ✓ Yes |
| Alternating speech/silence | α advances uniformly | α advances only during speech | ✓ Yes |
| Very short silence (<50ms) | α continues | Treated as speech (configurable) | ✓ Yes |
| Low SNR / noisy recording | α advances | May misclassify noise as speech | ⚠️ Tune threshold |

---

## Testing Strategy

### Unit Tests

```python
# tests/test_silence_aware_morph.py

def test_vad_all_speech():
    """VAD should mark all frames as speech if energy is high"""
    features = torch.randn(100, 1024)  # High energy
    is_speech = detect_voice_activity_energy(features, threshold_db=-40)
    assert is_speech.all()

def test_vad_all_silence():
    """VAD should mark all frames as silence if energy is low"""
    features = torch.randn(100, 1024) * 1e-6  # Very low energy
    is_speech = detect_voice_activity_energy(features, threshold_db=-40)
    assert not is_speech.any()

def test_silence_aware_alpha_frozen():
    """Alpha should freeze during silence"""
    is_speech = torch.tensor([1, 1, 0, 0, 1, 1], dtype=torch.bool)
    alpha = generate_morph_profile_silence_aware(6, is_speech, 'linear')

    # Silence frames should have same alpha as last speech frame
    assert alpha[2] == alpha[1]
    assert alpha[3] == alpha[1]

def test_silence_aware_alpha_advances():
    """Alpha should advance during speech"""
    is_speech = torch.tensor([1, 1, 0, 0, 1, 1], dtype=torch.bool)
    alpha = generate_morph_profile_silence_aware(6, is_speech, 'linear')

    # Alpha should increase during speech
    assert alpha[1] > alpha[0]
    assert alpha[5] > alpha[4]

def test_no_speech_edge_case():
    """All silence should result in alpha=0 everywhere"""
    is_speech = torch.zeros(100, dtype=torch.bool)
    alpha = generate_morph_profile_silence_aware(100, is_speech, 'linear')
    assert (alpha == 0).all()
```

### Integration Tests

```python
def test_match_morph_silence_aware():
    """End-to-end test of silence-aware morphing"""
    # Create synthetic query with silence
    query = torch.randn(100, 1024)
    query[40:60, :] *= 1e-6  # Insert silence

    match_a = torch.randn(500, 1024)
    match_b = torch.randn(500, 1024)

    # Test standard morphing
    out_std = knn_vc.match_morph(query, match_a, match_b, silence_aware=False)

    # Test silence-aware morphing
    out_sa = knn_vc.match_morph(query, match_a, match_b, silence_aware=True)

    # Both should produce valid output
    assert out_std.shape[0] > 0
    assert out_sa.shape[0] > 0

    # Outputs should differ (silence-aware modifies alpha trajectory)
    assert not torch.allclose(out_std, out_sa)
```

---

## Potential Pitfalls and Mitigations

### Pitfall 1: Over-Aggressive VAD

**Problem:** VAD marks too much as silence (e.g., quiet phonemes, fricatives)

**Symptoms:**
- Choppy morphing
- Alpha appears to "jump" frequently
- Morphing doesn't complete (ends at α < 1.0)

**Mitigation:**
- Lower threshold: `-50 dB` instead of `-40 dB`
- Increase `vad_min_silence_ms` to ignore brief dips
- Provide visualization tool to inspect VAD output

---

### Pitfall 2: Under-Aggressive VAD

**Problem:** VAD marks pauses as speech (e.g., background noise)

**Symptoms:**
- Silence-aware behaves identically to standard
- No visible alpha freezing
- Defeats the purpose of the feature

**Mitigation:**
- Raise threshold: `-30 dB` instead of `-40 dB`
- Use Silero VAD (more accurate)
- Document typical threshold ranges for different recording conditions

---

### Pitfall 3: Misalignment Between Features and Waveform

**Problem:** If using waveform-based VAD (Silero), frame boundaries may not align perfectly with WavLM features

**Symptoms:**
- Off-by-one errors in alpha freezing
- VAD decisions don't match perceived speech/silence

**Mitigation:**
- Use energy-based VAD on features (perfect alignment)
- If using Silero, carefully round timestamps to feature frames
- Add unit test to verify alignment

---

### Pitfall 4: User Confusion About Alpha Not Reaching 1.0

**Problem:** If final frames are silence, alpha may freeze before reaching 1.0

**Example:**
```
[Speech] [Speech] [Silence at end]
 α=0.5    α=0.8     α=0.8 (frozen)
                    ↑ never reaches 1.0
```

**Mitigation:**
- Document this behavior clearly
- Consider option to "force alpha to 1.0 at end" (post-processing)
- Recommend trimming silence from end before morphing

---

## Recommendations and Conclusion

### ✅ Core Recommendation: Implement the Feature

**Verdict:** The silence-aware morphing feature is **well-motivated and feasible**.

**Strengths:**
- Addresses real perceptual issue (morphing during silence)
- Clean conceptual model (morph only during speech)
- Backward compatible (optional parameter)
- Leverages existing VAD concepts in codebase

**Risks:**
- VAD accuracy (mitigated by tunable thresholds)
- User confusion about alpha trajectory (mitigated by documentation)
- Edge cases (mostly handled gracefully)

---

### Recommended Implementation Approach

**Start Simple:**
1. Energy-based VAD (fast, good enough)
2. Support all morph profiles (linear, sigmoid, etc.)
3. Tunable threshold and minimum silence duration
4. Clear documentation with examples

**Add Later if Needed:**
5. Silero VAD for higher accuracy
6. Adaptive thresholding
7. Visualization tools

---

### Estimated Effort

| Phase | Time Estimate |
|-------|---------------|
| Phase 1: Core implementation | 2-3 hours |
| Phase 2: Testing and validation | 2-3 hours |
| Phase 3: Documentation | 1-2 hours |
| **Total (V1)** | **5-8 hours** |
| Phase 4: Enhancements (optional) | +3-5 hours |

---

### Success Metrics

**Objective:**
- [ ] VAD correctly identifies speech vs. silence (>90% frame-level accuracy)
- [ ] Alpha frozen during silence spans (verifiable in plots)
- [ ] Alpha reaches 1.0 by end of final speech segment
- [ ] Backward compatible (existing tests pass with `silence_aware=False`)

**Subjective:**
- [ ] Silence-aware morphing sounds more natural on utterances with pauses
- [ ] Transitions at silence boundaries are smooth (no artifacts)
- [ ] Feature is easy to use (good defaults, clear documentation)

---

## Implementation Priority

**Immediate (Do First):**
1. Implement energy-based VAD
2. Implement silence-aware profile generation
3. Integrate with match_morph()
4. Unit tests for core functionality

**Short-term (Do This Week):**
5. Create visualization script
6. Test on real samples
7. Tune default parameters
8. Document in README_MORPHING.md

**Future (If Needed):**
9. Add Silero VAD
10. Adaptive thresholding
11. CLI tool with --silence-aware flag

---

## Open Questions for User

1. **Default behavior:** Should `silence_aware=False` or `True` by default?
   - **Recommendation:** `False` for backward compatibility, let users opt-in

2. **Threshold tuning:** What recording conditions will be tested?
   - Clean studio recordings: `-40 dB` works well
   - Noisy recordings: May need `-30 dB` or Silero VAD

3. **Minimum silence duration:** What's too short to consider?
   - **Recommendation:** 50ms (filters brief pauses between syllables)

4. **Force alpha to 1.0:** Should we ensure alpha reaches 1.0 at the end?
   - **Recommendation:** No, keep it pure (document the behavior)

5. **Integration with artifact fixes:** Should silence-aware morphing be combined with temporal smoothing?
   - **Recommendation:** Yes, they're complementary (smoothing helps at all boundaries)

---

## Final Verdict

**Implement the feature.** It's a clean, well-motivated enhancement that addresses a real perceptual issue. Start with energy-based VAD for simplicity, add Silero later if needed. The implementation is straightforward and the risks are manageable with good defaults and documentation.

**Estimated timeline:** 1-2 days for full implementation, testing, and documentation.
