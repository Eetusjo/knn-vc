"""
Simple Continuous Morphing Example

This is a minimal example showing how to use continuous morphing with kNN-VC.
Adapt the file paths to your own audio files.

NOTE: This uses the LOCAL implementation. To use torch.hub (remote repo),
you'll need to wait for the morphing feature to be merged upstream.
"""

import sys
from pathlib import Path

# Add parent directory to path to import local knn-vc modules
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torchaudio
from hubconf import knn_vc as load_knn_vc

# Step 1: Load the kNN-VC model from LOCAL implementation
print("Loading kNN-VC model from local code...")
knn_vc = load_knn_vc(prematched=True, pretrained=True, device='cpu')

# Step 2: Specify your audio files
# Replace these with your actual file paths
test_id = 0
ref_minutes = 10

source = "nainen"
target = "mies"
longsrc = True

#src_wav_path = f'sample_data/{source}/test/{source}_test{"_20s" if longsrc else ""}_{test_id}.wav'
#src_wav_path = f'sample_data/{source}/test/{source}_test_{test_id}.wav'
#src_wav_paths = [f"sample_data/japping2/japping_ref_{i}.wav" for i in range(5)]
src_wav_paths = ["sample_data/Recording.wav"]
#ref_A_paths = [f'sample_data/{source}/ref/{source}_ref_{i}.wav' for i in range(ref_minutes)]
#ref_B_paths = [f'sample_data/{target}/ref/{target}_ref_{i}.wav' for i in range(ref_minutes)]
ref_A_paths = [f"sample_data/japping2/japping_ref_{i}.wav" for i in range(5)]
ref_B_paths = ["sample_data/sv-speech.wav", "sample_data/sv-nonsense.wav"]


# Step 3: Extract features
print("Extracting features...")
matching_set_A = knn_vc.get_matching_set(ref_A_paths)
matching_set_B = knn_vc.get_matching_set(ref_B_paths)

# Step 4: Perform continuous morphing
print("Performing morphing...")
for i, src_wav_path in enumerate(src_wav_paths):
    query_seq = knn_vc.get_features(src_wav_path)
    #out_wav = knn_vc.match_morph(
    #    query_seq,
    #    matching_set_A,
    #    matching_set_B,
    #    silence_aware=True,
    #    vad_threshold_db=-30,
    #    topk=10,
    #    query_wav_path=src_wav_path,
    #    morph_profile='linear'
    #)
    out_wav = knn_vc.match(query_seq, matching_set_B, topk=8)
    # Step 5: Save the result
    morph_fname = f'morphed_{i}.wav'
    torchaudio.save(morph_fname, out_wav[None], 16000)
    print(f"Done! Output saved to: {morph_fname}")

# ============================================================================
# Tips:
# ============================================================================
# 1. Try different morph profiles:
#    - 'linear': constant rate of change
#    - 'sigmoid': smooth S-curve (more natural)
#    - 'step': abrupt change at midpoint (control condition)
#
# 2. Adjust sigmoid steepness:
#    out_wav = knn_vc.match_morph(..., morph_profile='sigmoid',
#                                 morph_params={'steepness': 20})
#
# 3. Use more reference audio for better quality (up to ~5 minutes per speaker)
#
# 4. Experiment with k (topk parameter):
#    - Lower k (e.g., 2): more source speaker characteristics preserved
#    - Higher k (e.g., 8): smoother but possibly less distinct
# ============================================================================
