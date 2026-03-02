#!/usr/bin/env python3
"""
Quick test to verify local implementation can be imported and has match_morph.
Run this to confirm everything is working.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("Testing Local kNN-VC Implementation")
print("="*60)

try:
    print("\n[1/4] Importing local modules...")
    from matcher import KNeighborsVC, generate_morph_profile
    print("  ✓ Successfully imported matcher module")

    from hubconf import knn_vc
    print("  ✓ Successfully imported hubconf module")

except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

try:
    print("\n[2/4] Testing generate_morph_profile()...")
    alpha = generate_morph_profile(100, 'linear')
    assert alpha.shape[0] == 100
    assert alpha[0] == 0.0
    assert alpha[-1] == 1.0
    print("  ✓ Linear profile works correctly")

    alpha = generate_morph_profile(100, 'sigmoid')
    assert alpha.shape[0] == 100
    print("  ✓ Sigmoid profile works correctly")

except Exception as e:
    print(f"  ✗ Profile generation failed: {e}")
    sys.exit(1)

try:
    print("\n[3/4] Checking KNeighborsVC has match_morph method...")
    assert hasattr(KNeighborsVC, 'match_morph')
    print("  ✓ match_morph method exists in KNeighborsVC class")

except AssertionError:
    print("  ✗ match_morph method not found")
    sys.exit(1)

try:
    print("\n[4/4] Verifying method signature...")
    import inspect
    sig = inspect.signature(KNeighborsVC.match_morph)
    params = list(sig.parameters.keys())

    expected_params = ['self', 'query_seq', 'matching_set_A', 'matching_set_B']
    assert all(p in params for p in expected_params)
    print("  ✓ match_morph has correct parameters")
    print(f"     Parameters: {', '.join(params[:8])}...")

except Exception as e:
    print(f"  ✗ Signature check failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("All Tests Passed! ✓")
print("="*60)
print("\nYour local implementation is ready to use.")
print("\nNext steps:")
print("  1. See USAGE_NOTES.md for import examples")
print("  2. Check examples/ directory for working scripts")
print("  3. Read README_MORPHING.md for full documentation")
print("\nTo use in your code:")
print("  from hubconf import knn_vc")
print("  model = knn_vc(pretrained=True, device='cuda')")
print("  # model.match_morph(...) is now available!")
print("="*60)
