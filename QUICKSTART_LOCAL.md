# Quick Start: Using Local Continuous Morphing

## TL;DR

```python
import sys
sys.path.insert(0, '/path/to/knn-vc')  # Point to this repo

from hubconf import knn_vc
import torchaudio

# Load model from LOCAL code (not torch.hub!)
model = knn_vc(pretrained=True, prematched=True, device='cuda')

# Extract features
query = model.get_features('source.wav')
match_a = model.get_matching_set(['speaker_a.wav'])
match_b = model.get_matching_set(['speaker_b.wav'])

# Morph from A to B
out = model.match_morph(query, match_a, match_b, topk=4)

# Save
torchaudio.save('morphed.wav', out[None], 16000)
```

## Why Not torch.hub.load()?

**Problem**: `torch.hub.load('bshall/knn-vc', ...)` uses the **remote** GitHub repo, which doesn't have `match_morph()` yet.

**Solution**: Import from your **local** code as shown above.

## Three Ways to Use Local Code

### 1. Direct Import (Simplest)

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/path/to/knn-vc')))

from hubconf import knn_vc
model = knn_vc(pretrained=True, device='cuda')
```

### 2. Run Examples (Zero Setup)

```bash
cd /path/to/knn-vc
python examples/simple_morph_example.py
```

Examples handle paths automatically.

### 3. Editable Install (For Development)

```bash
cd /path/to/knn-vc
pip install -e .
```

Now import works everywhere.

## Verify It Works

```bash
python test_local_import.py
```

Should print "All Tests Passed! ✓"

## Complete Example

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/path/to/knn-vc')  # UPDATE THIS PATH

from hubconf import knn_vc
import torchaudio

print("Loading model...")
model = knn_vc(pretrained=True, prematched=True, device='cuda')

print("Extracting features...")
query = model.get_features('test_source.wav')
match_a = model.get_matching_set(['speaker_a_ref1.wav', 'speaker_a_ref2.wav'])
match_b = model.get_matching_set(['speaker_b_ref1.wav', 'speaker_b_ref2.wav'])

print("Morphing from A to B...")
out = model.match_morph(
    query,
    match_a,
    match_b,
    topk=4,
    morph_profile='linear'  # Try: 'sigmoid' for smoother transition
)

print("Saving output...")
torchaudio.save('morphed_output.wav', out[None], 16000)
print("Done! Output: morphed_output.wav")
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: ... no attribute 'match_morph'` | Using torch.hub | Use local import |
| `ModuleNotFoundError: No module named 'matcher'` | Wrong path | Check sys.path points to knn-vc dir |
| `ImportError: cannot import name 'knn_vc'` | Not the right directory | Path should contain hubconf.py |

## See Also

- **USAGE_NOTES.md** - Detailed explanation of torch.hub vs. local
- **README_MORPHING.md** - Complete API documentation
- **examples/** - Working example scripts
- **test_local_import.py** - Verify your setup
