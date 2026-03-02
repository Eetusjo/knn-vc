# Using the Local Continuous Morphing Implementation

## Important: torch.hub vs. Local Code

When you use `torch.hub.load('bshall/knn-vc', 'knn_vc')`, PyTorch downloads and uses code from the **remote GitHub repository** (https://github.com/bshall/knn-vc), **NOT your local code**.

Since the continuous morphing feature (`match_morph()`) is only in your local implementation on the `es_continuous_morphing` branch, you need to load the model differently.

## Option 1: Import from Local hubconf (Recommended)

```python
import sys
from pathlib import Path

# Add knn-vc directory to Python path
sys.path.insert(0, '/path/to/knn-vc')  # Update this path

import torch
import torchaudio
from hubconf import knn_vc

# Load model from local code
model = knn_vc(pretrained=True, prematched=True, device='cuda')

# Now match_morph is available!
query_seq = model.get_features('source.wav')
matching_A = model.get_matching_set(['speaker_A.wav'])
matching_B = model.get_matching_set(['speaker_B.wav'])

out_wav = model.match_morph(query_seq, matching_A, matching_B, topk=4)
torchaudio.save('morphed.wav', out_wav[None], 16000)
```

## Option 2: Run Examples from the Repo Directory

The example scripts automatically handle the path setup:

```bash
cd /path/to/knn-vc

# Simple example
python examples/simple_morph_example.py

# Full CLI tool
python examples/continuous_morph_demo.py \
    --source your_source.wav \
    --speaker-a speaker_a_ref.wav \
    --speaker-b speaker_b_ref.wav \
    --output morphed.wav
```

## Option 3: Install Package in Editable Mode

Install your local version as a package:

```bash
cd /path/to/knn-vc
pip install -e .
```

Then you can import it anywhere:

```python
from matcher import KNeighborsVC
from hubconf import wavlm_large, hifigan_wavlm

# Load models
wavlm = wavlm_large(pretrained=True, device='cuda')
hifigan, hifigan_cfg = hifigan_wavlm(pretrained=True, prematched=True, device='cuda')

# Create kNN-VC instance with local code
knn_vc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, device='cuda')

# Use match_morph
out_wav = knn_vc.match_morph(...)
```

## Verifying You're Using Local Code

```python
# Check that match_morph exists
print(hasattr(knn_vc, 'match_morph'))  # Should print: True

# View the docstring
help(knn_vc.match_morph)

# Check available methods
print([m for m in dir(knn_vc) if not m.startswith('_')])
# Should include: 'match', 'match_morph', 'get_features', 'get_matching_set', 'vocode'
```

## Why This Happens

`torch.hub.load()` is designed to load models from GitHub repositories. When you call:

```python
torch.hub.load('bshall/knn-vc', 'knn_vc', ...)
```

PyTorch:
1. Downloads code from https://github.com/bshall/knn-vc
2. Caches it in `~/.cache/torch/hub/`
3. Imports the `knn_vc()` function from the **remote** `hubconf.py`
4. Returns the model instance

Your local changes are ignored completely.

## Alternative: Wait for Upstream Merge

Once the continuous morphing feature is:
1. Merged into the main knn-vc repository
2. Released by the original authors

Then `torch.hub.load('bshall/knn-vc', 'knn_vc')` will automatically include `match_morph()`.

Until then, use the local import methods described above.

## Quick Reference

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| `sys.path + import hubconf` | Quick scripts | Simple, no install needed | Requires path management |
| Run from repo directory | Using examples | Examples already set up | Must run from repo |
| `pip install -e .` | Development work | Works everywhere | Requires package setup |
| `torch.hub.load()` | After upstream merge | Clean, standard | Not available yet |

## Example: Complete Working Script

```python
#!/usr/bin/env python3
"""
Working example using local implementation
Save as: test_local_morph.py
Run as: python test_local_morph.py
"""

import sys
from pathlib import Path

# Find the knn-vc repository directory
# Assuming this script is in the same directory or subdirectory
repo_root = Path(__file__).parent.parent / 'knn-vc'  # Adjust if needed
sys.path.insert(0, str(repo_root))

import torch
import torchaudio
from hubconf import knn_vc

# Load model
print("Loading model from local implementation...")
model = knn_vc(pretrained=True, prematched=True, device='cuda')

# Verify match_morph exists
assert hasattr(model, 'match_morph'), "match_morph not found!"
print("✓ match_morph is available")

# Rest of your code...
print("\nReady to use continuous morphing!")
```

## Troubleshooting

**Problem**: `AttributeError: 'KNeighborsVC' object has no attribute 'match_morph'`
- **Cause**: Using torch.hub.load() which loads remote code
- **Solution**: Use local import as shown above

**Problem**: `ModuleNotFoundError: No module named 'matcher'`
- **Cause**: Python can't find your local knn-vc code
- **Solution**: Add the repo to sys.path (see examples above)

**Problem**: `ImportError: cannot import name 'knn_vc'`
- **Cause**: Wrong directory in sys.path
- **Solution**: Ensure path points to the directory containing hubconf.py

## See Also

- `examples/local_morph_example.py` - Demonstrates local import
- `examples/simple_morph_example.py` - Updated to use local code
- `examples/continuous_morph_demo.py` - CLI tool using local code
