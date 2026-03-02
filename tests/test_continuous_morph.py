"""
Unit tests for continuous morphing functionality.
"""

import torch
import pytest
from matcher import generate_morph_profile


class TestMorphProfiles:
    """Test suite for morph profile generation."""

    def test_linear_profile_boundaries(self):
        """Linear profile should start at 0 and end at 1."""
        alpha = generate_morph_profile(100, profile='linear')
        assert torch.isclose(alpha[0], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(alpha[-1], torch.tensor(1.0), atol=1e-6)

    def test_linear_profile_midpoint(self):
        """Linear profile should be 0.5 at the midpoint."""
        alpha = generate_morph_profile(100, profile='linear')
        # For 100 frames, midpoint is at index 49 (0-indexed)
        # Value should be close to 0.5
        assert torch.isclose(alpha[49], torch.tensor(0.5), atol=0.02)

    def test_linear_profile_monotonic(self):
        """Linear profile should be strictly increasing."""
        alpha = generate_morph_profile(100, profile='linear')
        diffs = alpha[1:] - alpha[:-1]
        assert torch.all(diffs > 0), "Linear profile should be strictly increasing"

    def test_sigmoid_profile_boundaries(self):
        """Sigmoid profile should start near 0 and end near 1."""
        alpha = generate_morph_profile(100, profile='sigmoid')
        assert alpha[0] < 0.1, "Sigmoid should start near 0"
        assert alpha[-1] > 0.9, "Sigmoid should end near 1"

    def test_sigmoid_profile_monotonic(self):
        """Sigmoid profile should be monotonically increasing."""
        alpha = generate_morph_profile(100, profile='sigmoid')
        diffs = alpha[1:] - alpha[:-1]
        assert torch.all(diffs >= 0), "Sigmoid profile should be monotonically increasing"

    def test_sigmoid_steepness_parameter(self):
        """Higher steepness should create sharper transitions."""
        alpha_gentle = generate_morph_profile(100, profile='sigmoid', params={'steepness': 5})
        alpha_steep = generate_morph_profile(100, profile='sigmoid', params={'steepness': 20})

        # At the midpoint, gentle curve should be closer to 0.5 than steep curve
        mid_idx = 50
        assert torch.abs(alpha_gentle[mid_idx] - 0.5) < torch.abs(alpha_steep[mid_idx] - 0.5)

    def test_step_profile_default_threshold(self):
        """Step profile should jump at threshold (default 0.5)."""
        alpha = generate_morph_profile(100, profile='step')

        # Before midpoint should be 0
        assert torch.all(alpha[:50] == 0.0), "First half should be 0"
        # After midpoint should be 1
        assert torch.all(alpha[50:] == 1.0), "Second half should be 1"

    def test_step_profile_custom_threshold(self):
        """Step profile should respect custom threshold."""
        alpha = generate_morph_profile(100, profile='step', params={'threshold': 0.3})

        # Before 30% should be 0
        assert torch.all(alpha[:30] == 0.0)
        # After 30% should be 1
        assert torch.all(alpha[30:] == 1.0)

    def test_custom_profile_valid(self):
        """Custom profile should accept user-provided values."""
        custom_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        alpha = generate_morph_profile(5, profile='custom', params={'alpha_values': custom_values})

        assert alpha.shape[0] == 5
        assert torch.allclose(alpha, torch.tensor(custom_values))

    def test_custom_profile_wrong_length_raises_error(self):
        """Custom profile should raise error if length doesn't match."""
        custom_values = [0.0, 0.5, 1.0]

        with pytest.raises(ValueError, match="must have 100 values"):
            generate_morph_profile(100, profile='custom', params={'alpha_values': custom_values})

    def test_custom_profile_missing_values_raises_error(self):
        """Custom profile should raise error if alpha_values not provided."""
        with pytest.raises(ValueError, match="requires 'alpha_values'"):
            generate_morph_profile(100, profile='custom', params={})

    def test_unknown_profile_raises_error(self):
        """Unknown profile type should raise error."""
        with pytest.raises(ValueError, match="Unknown morph profile"):
            generate_morph_profile(100, profile='nonexistent')

    def test_all_profiles_output_shape(self):
        """All profiles should output correct shape."""
        n_frames = 200
        for profile in ['linear', 'sigmoid', 'step']:
            alpha = generate_morph_profile(n_frames, profile=profile)
            assert alpha.shape == (n_frames,), f"Profile {profile} has wrong shape"

    def test_all_profiles_valid_range(self):
        """All profiles should output values in [0, 1]."""
        n_frames = 200
        for profile in ['linear', 'sigmoid', 'step']:
            alpha = generate_morph_profile(n_frames, profile=profile)
            assert torch.all(alpha >= 0.0), f"Profile {profile} has values < 0"
            assert torch.all(alpha <= 1.0), f"Profile {profile} has values > 1"

    def test_short_sequences(self):
        """Profiles should work with very short sequences."""
        for profile in ['linear', 'sigmoid', 'step']:
            alpha = generate_morph_profile(2, profile=profile)
            assert alpha.shape == (2,)
            assert alpha[0] < alpha[1] or profile == 'step'

    def test_long_sequences(self):
        """Profiles should work with long sequences."""
        n_frames = 10000
        for profile in ['linear', 'sigmoid', 'step']:
            alpha = generate_morph_profile(n_frames, profile=profile)
            assert alpha.shape == (n_frames,)


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
