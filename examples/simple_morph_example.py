import sys
from pathlib import Path

# Add parent directory to path to import local knn-vc modules
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import torch
import torchaudio
from hubconf import knn_vc as load_knn_vc

# Step 1: Load the kNN-VC model
print("Loading kNN-VC model from local code...")
knn_vc = load_knn_vc(prematched=True, pretrained=True, device='cuda')

# Step 2: Specify your audio files
# Replace these with your actual file paths
src_wav_path = 'path/to/source.wav'  # The speech content to morph
ref_A_paths = ['path/to/speaker_A_ref1.wav', 'path/to/speaker_A_ref2.wav']  # Start speaker
ref_B_paths = ['path/to/speaker_B_ref1.wav', 'path/to/speaker_B_ref2.wav']  # End speaker

# Step 3: Extract features
print("Extracting features")
query_seq = knn_vc.get_features(src_wav_path)
matching_set_A = knn_vc.get_matching_set(ref_A_paths)
matching_set_B = knn_vc.get_matching_set(ref_B_paths)

# Step 4: Perform continuous morphing
print("Performing morphing")
out_wav = knn_vc.match_morph(
    query_seq,
    matching_set_A,
    matching_set_B,
    topk=4,
    morph_profile='linear'  # 'linear', 'sigmoid', 'step'
)

# Step 5: Save the result
print("Saving output")
torchaudio.save('morphed_output.wav', out_wav[None], 16000)

print("Output saved to: morphed_output.wav")
