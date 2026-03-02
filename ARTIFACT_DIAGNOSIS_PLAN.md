# Artifact Diagnosis and Mitigation Plan for kNN-VC Continuous Morphing

## Problem Statement

User reports hearing "sudden spikes in pitch or something every once in a while" during continuous morphing. Artifacts are:
- Not consistent across test samples
- Described as pitch spikes or sudden changes
- Occur intermittently during morphing

## Research Summary

Based on detailed codebase analysis, the artifacts are most likely caused by **temporal discontinuities** in the frame-independent kNN matching process, amplified by the lack of temporal smoothing.

---

## Root Causes (Ranked by Probability)

### 1. Temporal Discontinuity in kNN Matching ⚠️ HIGH PRIORITY

**Mechanism:**
- Each frame independently finds k-nearest neighbors in both Speaker A and B reference sets
- Adjacent frames can match **completely different** reference frames with different pitch characteristics
- Even with smooth alpha blending, the underlying matched features can jump discontinuously
- The vocoder tries to reconstruct smooth audio from these discontinuous feature transitions

**Evidence from code:**
```python
# matcher.py lines 299-308
dists_A = fast_cosine_dist(query_seq, matching_set_A, device=device)
dists_B = fast_cosine_dist(query_seq, matching_set_B, device=device)

best_A = dists_A.topk(k=topk, largest=False, dim=-1)  # Independent per frame
best_B = dists_B.topk(k=topk, largest=False, dim=-1)  # No temporal consistency

out_feats_A = synth_set_A[best_A.indices].mean(dim=1)
out_feats_B = synth_set_B[best_B.indices].mean(dim=1)
```

**Why this causes pitch spikes:**
```
Example scenario:
Frame t:   matches ref frames [100, 102, 105, 108] (speaker in low pitch region)
Frame t+1: matches ref frames [523, 525, 530, 531] (speaker in high pitch region)
           ↓
Even with smooth alpha, the blended features jump from low-F0 to high-F0
           ↓
Vocoder creates pitch spike when reconstructing discontinuous features
```

### 2. Feature Manifold Violation ⚠️ MEDIUM-HIGH PRIORITY

**Mechanism:**
- Linear interpolation `(1-α)*A + α*B` in WavLM feature space may create "impossible" features
- These out-of-distribution features don't lie on the natural speech manifold
- The prematched HiFiGAN vocoder is trained on real kNN-matched features, NOT on blended features
- Vocoder encounters unexpected inputs → produces artifacts

**Evidence:**
- No normalization before/after blending (line 315)
- Simple linear blend assumes convex feature space
- Prematched vocoder has never seen blended features during training

### 3. Insufficient kNN Averaging ⚠️ MEDIUM PRIORITY

**Mechanism:**
- Default k=4 may be too small to smooth frame-to-frame variations
- Lower k → more sensitive to individual reference frame selection
- More variance in matched features → more discontinuities

**Current defaults:**
- `match_morph()`: topk=4 (line 235)
- `match()`: topk=4 (line 188)

**README suggests:** "Higher values (6-8): Smoother output"

### 4. HiFiGAN Transposed Convolution Artifacts ⚠️ LOW-MEDIUM PRIORITY

**Mechanism:**
- Transposed convolutions can amplify input discontinuities
- Upsampling configuration `[10, 8, 2, 2]` creates 80x intermediate upsampling
- No anti-aliasing between upsample stages
- When input features jump, upsampling can create pitch spikes

**HiFiGAN configuration:**
```python
# hifigan/models.py
upsample_rates = [10, 8, 2, 2]  # Total: 320x upsampling
hop_length = 320  # 20ms frames
```

---

## Critical Differences: match_morph() vs. match()

| Aspect | match() | match_morph() | Impact |
|--------|---------|---------------|--------|
| kNN lookups | 1x per frame | **2x per frame** | Double opportunity for discontinuities |
| Feature processing | Direct kNN average | **Linear blend of two kNN averages** | Can create out-of-distribution features |
| Temporal smoothing | None | **None** | Same lack of smoothing despite 2x complexity |
| Feature normalization | None | **None** | Blended features may have wrong magnitude |
| Default k | 4 | **4** | Same k despite higher variance |

**Key insight:** `match_morph()` has inherently more sources of discontinuity but applies the same (lack of) smoothing as `match()`.

---

## Diagnostic Experiments

### Phase 1: Identify the Root Cause

#### Experiment 1.1: Visualize Feature Discontinuities 🔬 CRITICAL

**Goal:** Determine if matched features have frame-to-frame jumps

**Implementation:**
```python
import matplotlib.pyplot as plt
import torch

# Inside match_morph(), after line 308, add:
out_feats_A = synth_set_A[best_A.indices].mean(dim=1)
out_feats_B = synth_set_B[best_B.indices].mean(dim=1)

# Compute frame-to-frame L2 distances
diff_A = torch.diff(out_feats_A, dim=0).norm(dim=1).cpu()
diff_B = torch.diff(out_feats_B, dim=0).norm(dim=1).cpu()
diff_blend = torch.diff(out_feats, dim=0).norm(dim=1).cpu()

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
axes[0].plot(diff_A)
axes[0].set_title('Speaker A: Frame-to-Frame Feature Distance')
axes[1].plot(diff_B)
axes[1].set_title('Speaker B: Frame-to-Frame Feature Distance')
axes[2].plot(diff_blend)
axes[2].set_title('Blended: Frame-to-Frame Feature Distance')
plt.savefig('feature_discontinuities.png')
```

**Hypothesis:** Spikes in the plot correlate with audible pitch artifacts

**What to look for:**
- Large spikes (>2.0 L2 distance) indicate discontinuities
- Compare spike locations with listening tests
- Check if spikes are more frequent in blended vs. single-speaker

**Expected outcome:**
- If spikes correlate with artifacts → Root Cause #1 confirmed
- If no correlation → Look at Root Cause #2 or #4

---

#### Experiment 1.2: Test with Constant Alpha 🔬 CRITICAL

**Goal:** Determine if the alpha interpolation curve causes artifacts

**Implementation:**
```python
# Create a test script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hubconf import knn_vc
import torch

knn_vc = knn_vc(pretrained=True, prematched=True, device='cuda')

query_seq = knn_vc.get_features('sample_data/mies/test/mies_test_0.wav')
match_a = knn_vc.get_matching_set([f'sample_data/mies/ref/mies_ref_{i}.wav' for i in range(5)])
match_b = knn_vc.get_matching_set([f'sample_data/nainen/ref/nainen_ref_{i}.wav' for i in range(5)])

# Test 1: Standard morphing (linear alpha)
out_normal = knn_vc.match_morph(query_seq, match_a, match_b, topk=4, morph_profile='linear')

# Test 2: Constant alpha = 0.5 (equal blend throughout)
out_const = knn_vc.match_morph(query_seq, match_a, match_b, topk=4,
                                morph_profile='custom',
                                morph_params={'alpha_values': [0.5] * len(query_seq)})

# Save both and listen
import torchaudio
torchaudio.save('test_normal_morph.wav', out_normal[None], 16000)
torchaudio.save('test_constant_alpha.wav', out_const[None], 16000)
```

**What to listen for:**
- If constant alpha has NO artifacts → problem is alpha interpolation curve
- If constant alpha STILL has artifacts → problem is kNN discontinuity (Root Cause #1)

**Expected outcome:** Artifacts persist with constant alpha → confirms Root Cause #1

---

#### Experiment 1.3: Extract and Visualize Pitch Contours 🔬 HIGH PRIORITY

**Goal:** Directly measure pitch spikes in the output

**Implementation:**
```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load morphed output
y, sr = librosa.load('morphed_output.wav', sr=16000)

# Extract pitch
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=80, fmax=400, sr=sr)

# Identify sudden pitch jumps
f0_clean = f0[~np.isnan(f0)]
f0_diff = np.abs(np.diff(f0_clean))
spike_threshold = 50  # Hz
spike_frames = np.where(f0_diff > spike_threshold)[0]

print(f"Found {len(spike_frames)} pitch spikes > {spike_threshold} Hz")
print(f"Spike locations (seconds): {spike_frames * 0.02}")

# Visualize
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(f0, label='F0 contour')
plt.ylabel('Frequency (Hz)')
plt.title('Pitch Contour')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(f0_diff)
plt.axhline(y=spike_threshold, color='r', linestyle='--', label='Spike threshold')
plt.ylabel('F0 change (Hz)')
plt.xlabel('Frame')
plt.title('Frame-to-Frame Pitch Changes')
plt.legend()
plt.savefig('pitch_analysis.png')
```

**What to analyze:**
- Number and magnitude of pitch spikes
- Correlation with feature discontinuities (from Exp 1.1)
- Timing relative to alpha values (beginning, middle, end of morph)

---

#### Experiment 1.4: Test k Value Sensitivity 🔬 HIGH PRIORITY

**Goal:** Determine if increasing k reduces artifacts

**Implementation:**
```python
# Test different k values
k_values = [2, 4, 8, 12, 16]
outputs = {}

for k in k_values:
    out = knn_vc.match_morph(query_seq, match_a, match_b, topk=k)
    outputs[f'morph_k{k}.wav'] = out
    torchaudio.save(f'morph_k{k}.wav', out[None], 16000)

# Listen to all outputs and rate artifact severity
```

**What to listen for:**
- Does artifact severity decrease with higher k?
- Is there a "sweet spot" k value?
- Do artifacts disappear entirely at high k (e.g., k=16)?

**Expected outcome:**
- If artifacts reduce with higher k → Root Cause #3 (insufficient averaging)
- If artifacts persist even at k=16 → Root Cause #1 or #2 (deeper structural issue)

---

### Phase 2: Test Potential Fixes

#### Experiment 2.1: Apply Temporal Smoothing 🔧 HIGH PRIORITY

**Goal:** Test if smoothing the blended features eliminates artifacts

**Implementation:** Add to `matcher.py` after line 315:

```python
def smooth_features_temporal(features, kernel_size=5):
    """Apply temporal smoothing via 1D convolution"""
    import torch.nn.functional as F

    # Create averaging kernel
    kernel = torch.ones(1, 1, kernel_size, device=features.device) / kernel_size

    # Smooth each feature dimension independently
    # Input: (n_frames, feat_dim) → (1, feat_dim, n_frames)
    feats_transposed = features.T.unsqueeze(0)

    # Pad temporally to preserve length
    padding = kernel_size // 2
    feats_padded = F.pad(feats_transposed, (padding, padding), mode='replicate')

    # Apply 1D convolution for smoothing
    feats_smoothed = F.conv1d(feats_padded, kernel.expand(features.shape[1], 1, -1),
                              groups=features.shape[1])

    # Back to (n_frames, feat_dim)
    return feats_smoothed.squeeze(0).T

# After line 315 in match_morph()
out_feats = (1 - alpha[:, None]) * out_feats_A + alpha[:, None] * out_feats_B

# ADD THIS LINE:
out_feats = smooth_features_temporal(out_feats, kernel_size=5)

# Then continue with vocoding...
prediction = self.vocode(out_feats[None].to(device)).cpu().squeeze()
```

**Test different kernel sizes:**
- `kernel_size=3` (60ms smoothing)
- `kernel_size=5` (100ms smoothing) ← recommended start
- `kernel_size=7` (140ms smoothing)

**What to listen for:**
- Do artifacts disappear?
- Does speech quality remain good (not over-smoothed)?
- Is there a trade-off between artifact reduction and naturalness?

**Expected outcome:** Significant artifact reduction with minimal quality loss

---

#### Experiment 2.2: Feature Normalization Before/After Blending 🔧 MEDIUM PRIORITY

**Goal:** Test if normalizing features prevents manifold violations

**Implementation:** Modify line 307-315 in `matcher.py`:

```python
import torch.nn.functional as F

# Get matched features
out_feats_A = synth_set_A[best_A.indices].mean(dim=1)  # (n_frames, dim)
out_feats_B = synth_set_B[best_B.indices].mean(dim=1)  # (n_frames, dim)

# OPTION 1: Normalize before blending
out_feats_A_norm = F.normalize(out_feats_A, p=2, dim=1)
out_feats_B_norm = F.normalize(out_feats_B, p=2, dim=1)
out_feats = (1 - alpha[:, None]) * out_feats_A_norm + alpha[:, None] * out_feats_B_norm

# OPTION 2: Normalize after blending (probably better)
out_feats = (1 - alpha[:, None]) * out_feats_A + alpha[:, None] * out_feats_B
out_feats = F.normalize(out_feats, p=2, dim=1)

# OPTION 3: Both (most conservative)
out_feats_A_norm = F.normalize(out_feats_A, p=2, dim=1)
out_feats_B_norm = F.normalize(out_feats_B, p=2, dim=1)
out_feats = (1 - alpha[:, None]) * out_feats_A_norm + alpha[:, None] * out_feats_B_norm
out_feats = F.normalize(out_feats, p=2, dim=1)
```

**Test all three options** and compare artifacts

**Expected outcome:** Moderate improvement, especially combined with smoothing

---

#### Experiment 2.3: Temporal Consistency in kNN Matching 🔧 LOW PRIORITY (Complex)

**Goal:** Encourage consecutive frames to match nearby reference frames

**Implementation:** This requires significant refactoring. Sketch:

```python
# Replace independent topk with temporal-aware search
prev_indices_A = None
prev_indices_B = None
matched_indices_A = []
matched_indices_B = []

for t in range(n_frames):
    dists_A_t = dists_A[t].clone()
    dists_B_t = dists_B[t].clone()

    if prev_indices_A is not None:
        # Add bonus for reference frames near previous match
        # (Within ±10 frames in the reference set)
        for prev_idx in prev_indices_A:
            nearby_range = range(max(0, prev_idx-10), min(len(dists_A_t), prev_idx+10))
            dists_A_t[nearby_range] *= 0.9  # 10% bonus

    best_A = dists_A_t.topk(k=topk, largest=False)
    best_B = dists_B_t.topk(k=topk, largest=False)

    matched_indices_A.append(best_A.indices)
    matched_indices_B.append(best_B.indices)

    prev_indices_A = best_A.indices
    prev_indices_B = best_B.indices

# Then use matched_indices_A/B for feature extraction
```

**Note:** This is complex and may not be necessary if smoothing (Exp 2.1) works well

---

## Implementation Priority

### Immediate Actions (Do First)

1. **Run Experiment 1.1** (Feature discontinuity visualization)
   - Quick to implement
   - Provides hard evidence of discontinuities
   - Guides subsequent experiments

2. **Run Experiment 1.2** (Constant alpha test)
   - Minimal code change
   - Clearly separates alpha-curve issues from kNN issues
   - Fast diagnostic

3. **Run Experiment 1.4** (k value sensitivity)
   - Zero code change (just parameter sweep)
   - May reveal simple fix (increase default k)
   - Cheap to test

### High-Value Fixes (Likely to Work)

4. **Implement Experiment 2.1** (Temporal smoothing)
   - High probability of fixing artifacts
   - Low risk of degrading quality
   - Easy to implement and parameterize
   - Can be made optional (smoothing_kernel_size parameter)

5. **Increase default k** from 4 to 8 for `match_morph()`
   - Based on README recommendation
   - Zero downside (just slower)
   - May be sufficient alone or combined with smoothing

### Optional Enhancements (If Above Don't Fully Fix)

6. **Implement Experiment 2.2** (Feature normalization)
   - Moderate implementation complexity
   - May help with manifold violations
   - Combine with smoothing for best results

7. **Extract pitch contours** (Experiment 1.3)
   - More for analysis than fixing
   - Helps understand artifact characteristics
   - Good for documenting the fix

---

## Recommended Implementation Plan

### Step 1: Diagnostic Phase (1-2 hours)

Create a diagnostic script:

```bash
# examples/diagnose_artifacts.py
```

This script should:
1. Load a test sample
2. Run match_morph() with instrumentation
3. Extract and visualize:
   - Feature discontinuities
   - Pitch contours
   - Alpha profile
4. Test constant alpha vs. linear alpha
5. Test k=4 vs k=8 vs k=16
6. Save all plots and audio outputs

**Deliverable:** Clear visualization showing discontinuity spikes correlating with artifacts

---

### Step 2: Implement Smoothing Fix (1 hour)

Modify `matcher.py`:

1. Add `smooth_features_temporal()` helper function
2. Add optional `temporal_smoothing` parameter to `match_morph()`:
   ```python
   def match_morph(..., temporal_smoothing: int | None = 5):
       """
       ...
       temporal_smoothing: kernel size for temporal smoothing (None to disable)
           - 3: Light smoothing (60ms)
           - 5: Medium smoothing (100ms, recommended)
           - 7: Heavy smoothing (140ms)
       """
   ```
3. Apply smoothing after feature blending if enabled
4. Update defaults: k=4 → k=8, temporal_smoothing=5

**Deliverable:** Updated `match_morph()` with smoothing option

---

### Step 3: Validation Phase (1 hour)

Test the fix:

1. Re-run all test samples
2. Compare before/after:
   - Subjective listening (artifact count/severity)
   - Objective metrics (pitch contour smoothness)
   - Feature discontinuity plots
3. Tune parameters (k, smoothing kernel size)
4. Document optimal settings

**Deliverable:** Before/after comparison showing artifact reduction

---

### Step 4: Documentation (30 min)

Update documentation:

1. **README_MORPHING.md**: Add troubleshooting section on artifacts
2. Add parameter guidelines:
   - Default: `topk=8, temporal_smoothing=5`
   - If artifacts persist: increase k to 12-16
   - If over-smoothed: reduce smoothing to 3 or None
3. Update examples to use new defaults
4. Add note about artifact causes and fixes

---

## Expected Outcomes

### Best Case
- Temporal smoothing with k=8 eliminates most/all artifacts
- Minimal quality degradation
- Simple, optional parameter users can tune

### Likely Case
- Smoothing significantly reduces artifacts
- Some edge cases may need higher k or stronger smoothing
- Slight reduction in "crispness" due to smoothing
- Overall quality remains good

### Worst Case
- Artifacts partially reduced but not eliminated
- May need more complex temporal consistency approach (Exp 2.3)
- Could indicate vocoder limitation (need to retrain on blended features)

---

## Alternative Approaches (If Smoothing Doesn't Work)

### Plan B: Segment-Level Blending
Instead of frame-level blending, blend at phoneme/word segments:
- Reduces discontinuities by maintaining longer temporal context
- More complex implementation
- Requires phoneme alignment

### Plan C: Retrain Vocoder on Blended Features
- Fine-tune HiFiGAN on synthetically blended WavLM features
- Teaches vocoder to handle interpolated features
- Significant compute cost

### Plan D: Use Weighted kNN Instead of Blending
Instead of separate A/B matches + blend:
- Create single combined matching set with time-varying weights
- May reduce discontinuities by maintaining single kNN graph
- Different algorithm, more complex

---

## Success Metrics

### Objective
- [ ] Frame-to-frame feature distance reduced by >50%
- [ ] Pitch contour smoothness improved (fewer spikes >50Hz)
- [ ] Feature discontinuity spikes correlate <0.3 with perceived artifacts

### Subjective
- [ ] Artifacts "barely noticeable" or "not noticeable" in listening tests
- [ ] Morphing sounds smooth and natural
- [ ] No significant quality degradation vs. standard match()

---

## Next Steps

1. **Immediate:** Run diagnostic experiments to confirm root cause
2. **This week:** Implement temporal smoothing fix
3. **Validation:** Test on all sample files, tune parameters
4. **Documentation:** Update README and examples with findings

**Estimated total time:** 3-5 hours for full diagnostic → fix → validation cycle
