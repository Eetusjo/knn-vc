"""
Local Continuous Morphing Example

This example shows how to use the continuous morphing feature with your
LOCAL implementation (not torch.hub which uses the remote repo).
"""

import sys
from pathlib import Path

# Add the knn-vc directory to Python path so we can import local modules
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torchaudio
from hubconf import knn_vc  # Import from LOCAL hubconf.py

# Load the model using local code
print("Loading kNN-VC model from local implementation...")
model = knn_vc(pretrained=True, prematched=True, device='cuda')

# Verify the match_morph method exists
print(f"match_morph available: {hasattr(model, 'match_morph')}")
print(f"\nAvailable methods: {[m for m in dir(model) if not m.startswith('_')]}")

# Example usage (update paths to your actual audio files)
print("\n" + "="*60)
print("Example Usage:")
print("="*60)

# Specify your audio files
src_wav_path = 'path/to/source.wav'
ref_A_paths = ['path/to/speaker_A_ref.wav']
ref_B_paths = ['path/to/speaker_B_ref.wav']

print(f"""
# Extract features
query_seq = model.get_features('{src_wav_path}')
matching_A = model.get_matching_set({ref_A_paths})
matching_B = model.get_matching_set({ref_B_paths})

# Perform continuous morphing
out_wav = model.match_morph(
    query_seq,
    matching_A,
    matching_B,
    topk=4,
    morph_profile='linear'  # or 'sigmoid', 'step'
)

# Save result
torchaudio.save('morphed_output.wav', out_wav[None], 16000)
""")

print("="*60)
print("\nTips:")
print("  1. This script imports from your LOCAL code")
print("  2. Update the audio file paths above")
print("  3. Run from anywhere - it finds the repo automatically")
print("="*60)
