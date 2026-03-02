"""
Unit tests for continuous morphing functionality.
"""

import torch
import pytest
from matcher import (
    generate_morph_profile,
    generate_morph_profile_silence_aware,
    detect_voice_activity_energy,
    _filter_short_silences
)


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


class TestVoiceActivityDetection:
    """Test suite for energy-based VAD."""

    def test_vad_all_speech(self):
        """VAD should mark all frames as speech if energy is uniformly high."""
        # Create high-variance features (simulating speech)
        # Speech has dynamic content → high variance
        features = torch.randn(100, 1024) * 5.0  # High variance
        is_speech = detect_voice_activity_energy(features, threshold_db=-40)

        # Most frames should be detected as speech
        assert is_speech.sum() >= 90, f"Expected mostly speech, got {is_speech.sum()}/100"

    def test_vad_all_silence(self):
        """VAD should mark all frames as silence if variance is very low."""
        # Create constant/low-variance features (simulating silence)
        # Silence has little variation → low variance
        features = torch.ones(100, 1024) * 0.5 + torch.randn(100, 1024) * 1e-6
        is_speech = detect_voice_activity_energy(features, threshold_db=-40)

        # Most frames should be detected as silence
        assert is_speech.sum() <= 10, f"Expected mostly silence, got {is_speech.sum()} speech frames"

    def test_vad_mixed_content(self):
        """VAD should distinguish speech from silence in mixed audio."""
        features = torch.zeros(100, 1024)

        # First 30 frames: high variance (speech)
        features[0:30] = torch.randn(30, 1024) * 5.0

        # Middle 40 frames: low variance (silence)
        features[30:70] = torch.ones(40, 1024) * 0.5 + torch.randn(40, 1024) * 1e-6

        # Last 30 frames: high variance (speech)
        features[70:100] = torch.randn(30, 1024) * 5.0

        is_speech = detect_voice_activity_energy(features, threshold_db=-40)

        # Check that speech regions are detected
        assert is_speech[0:30].sum() >= 20, "First speech region not detected"
        assert is_speech[70:100].sum() >= 20, "Second speech region not detected"

        # Check that silence region is detected (may have some false positives at boundaries)
        assert is_speech[35:65].sum() <= 15, "Silence region incorrectly marked as speech"

    def test_vad_threshold_sensitivity(self):
        """Lower threshold should detect more frames as speech."""
        features = torch.randn(100, 1024) * 0.5  # Medium energy

        # Stricter threshold (higher value, less sensitive)
        is_speech_strict = detect_voice_activity_energy(features, threshold_db=-20)

        # Looser threshold (lower value, more sensitive)
        is_speech_loose = detect_voice_activity_energy(features, threshold_db=-50)

        # Loose threshold should detect more speech
        assert is_speech_loose.sum() >= is_speech_strict.sum()

    def test_vad_output_shape(self):
        """VAD should return boolean tensor with correct shape."""
        features = torch.randn(100, 1024)
        is_speech = detect_voice_activity_energy(features)

        assert is_speech.shape == (100,), f"Expected shape (100,), got {is_speech.shape}"
        assert is_speech.dtype == torch.bool, f"Expected bool dtype, got {is_speech.dtype}"

    def test_filter_short_silences(self):
        """Short silence spans should be filtered out."""
        # Create pattern: speech (10) → silence (3) → speech (10)
        is_speech = torch.tensor([True]*10 + [False]*3 + [True]*10)

        # Filter silences shorter than 5 frames
        filtered = _filter_short_silences(is_speech, min_frames=5)

        # The 3-frame silence should now be marked as speech
        assert filtered[10:13].all(), "Short silence not filtered"

    def test_filter_preserves_long_silences(self):
        """Long silence spans should be preserved."""
        # Create pattern: speech (10) → silence (10) → speech (10)
        is_speech = torch.tensor([True]*10 + [False]*10 + [True]*10)

        # Filter silences shorter than 5 frames
        filtered = _filter_short_silences(is_speech, min_frames=5)

        # The 10-frame silence should remain
        assert not filtered[10:20].any(), "Long silence incorrectly filtered"


class TestSilenceAwareMorphing:
    """Test suite for silence-aware morph profile generation."""

    def test_silence_aware_alpha_frozen_during_silence(self):
        """Alpha should freeze during silence spans."""
        # Pattern: speech (20) → silence (10) → speech (20)
        is_speech = torch.tensor([True]*20 + [False]*10 + [True]*20)

        alpha = generate_morph_profile_silence_aware(50, is_speech, 'linear')

        # During silence (frames 20-29), alpha should be constant
        silence_alpha = alpha[20:30]
        assert torch.all(silence_alpha == silence_alpha[0]), \
            "Alpha should be constant during silence"

        # Alpha at frame 20 (last speech before silence) should equal alpha during silence
        assert torch.isclose(alpha[19], alpha[25]), \
            "Silence alpha should equal last speech value"

    def test_silence_aware_alpha_advances_during_speech(self):
        """Alpha should advance during speech frames."""
        # Pattern: speech (20) → silence (10) → speech (20)
        is_speech = torch.tensor([True]*20 + [False]*10 + [True]*20)

        alpha = generate_morph_profile_silence_aware(50, is_speech, 'linear')

        # Alpha should increase during first speech segment
        assert alpha[19] > alpha[0], "Alpha should increase during speech"

        # Alpha should increase during second speech segment
        assert alpha[49] > alpha[30], "Alpha should increase during speech"

    def test_silence_aware_reaches_one(self):
        """Alpha should reach 1.0 at the final speech frame."""
        # Pattern: speech (30) → silence (20)
        is_speech = torch.tensor([True]*30 + [False]*20)

        alpha = generate_morph_profile_silence_aware(50, is_speech, 'linear')

        # Alpha should be 1.0 at last speech frame (index 29) and all subsequent frames
        assert torch.isclose(alpha[29], torch.tensor(1.0), atol=1e-6), \
            f"Alpha should be 1.0 at last speech frame, got {alpha[29]}"
        assert torch.all(alpha[29:] == 1.0), \
            "Alpha should remain 1.0 after last speech frame"

    def test_silence_aware_no_speech(self):
        """All silence should result in alpha=0 everywhere."""
        is_speech = torch.zeros(100, dtype=torch.bool)

        alpha = generate_morph_profile_silence_aware(100, is_speech, 'linear')

        assert torch.all(alpha == 0.0), \
            "All-silence input should produce alpha=0 everywhere"

    def test_silence_aware_no_silence(self):
        """No silence should behave like standard morphing."""
        is_speech = torch.ones(100, dtype=torch.bool)

        alpha_silence_aware = generate_morph_profile_silence_aware(100, is_speech, 'linear')
        alpha_standard = generate_morph_profile(100, 'linear')

        # Both should be very similar (minor differences due to implementation)
        assert torch.allclose(alpha_silence_aware, alpha_standard, atol=0.02), \
            "No-silence should behave like standard morphing"

    def test_silence_aware_starts_with_silence(self):
        """Silence at start should keep alpha at 0."""
        # Pattern: silence (20) → speech (30)
        is_speech = torch.tensor([False]*20 + [True]*30)

        alpha = generate_morph_profile_silence_aware(50, is_speech, 'linear')

        # Alpha should be 0 during initial silence
        assert torch.all(alpha[0:20] == 0.0), \
            "Alpha should be 0 during initial silence"

        # Alpha should advance during speech
        assert alpha[49] > alpha[20], \
            "Alpha should advance after silence ends"

    def test_silence_aware_ends_with_silence(self):
        """Silence at end should freeze alpha at last speech value."""
        # Pattern: speech (30) → silence (20)
        is_speech = torch.tensor([True]*30 + [False]*20)

        alpha = generate_morph_profile_silence_aware(50, is_speech, 'linear')

        # Alpha during final silence should equal 1.0 (last speech value)
        assert torch.all(alpha[30:] == 1.0), \
            "Alpha should be frozen at 1.0 during final silence"

    def test_silence_aware_sigmoid_profile(self):
        """Silence-aware should work with sigmoid profile."""
        # Pattern: speech (20) → silence (10) → speech (20)
        is_speech = torch.tensor([True]*20 + [False]*10 + [True]*20)

        alpha = generate_morph_profile_silence_aware(
            50, is_speech, 'sigmoid', {'steepness': 10}
        )

        # Check basic properties
        assert alpha[0] < alpha[49], "Alpha should increase overall"
        assert torch.all(alpha[20:30] == alpha[20]), "Alpha frozen during silence"
        assert torch.isclose(alpha[49], torch.tensor(1.0), atol=0.05), \
            "Alpha should reach ~1.0 at end"

    def test_silence_aware_step_profile(self):
        """Silence-aware should work with step profile."""
        # Pattern: speech (25) → silence (10) → speech (25)
        is_speech = torch.tensor([True]*25 + [False]*10 + [True]*25)

        alpha = generate_morph_profile_silence_aware(
            60, is_speech, 'step', {'threshold': 0.5}
        )

        # Step should transition at 50% of speech content (25 speech frames)
        # First 25 speech frames: α≈0
        # Next 25 speech frames: α≈1
        assert alpha[24] < 0.5, "First half of speech should have α<0.5"
        assert alpha[59] > 0.5, "Second half of speech should have α>0.5"

    def test_silence_aware_alternating_pattern(self):
        """Complex speech/silence pattern should be handled correctly."""
        # Pattern: S(10) Si(5) S(10) Si(5) S(10) Si(5) S(10) Si(5)
        is_speech = torch.cat([
            torch.tensor([True]*10 + [False]*5) for _ in range(4)
        ])

        alpha = generate_morph_profile_silence_aware(60, is_speech, 'linear')

        # Alpha should be frozen during each silence segment
        for i in range(4):
            silence_start = 10 + i * 15
            silence_end = silence_start + 5
            silence_alpha = alpha[silence_start:silence_end]

            assert torch.all(silence_alpha == silence_alpha[0]), \
                f"Alpha not frozen during silence segment {i}"

    def test_silence_aware_output_range(self):
        """Alpha should always be in [0, 1] range."""
        is_speech = torch.tensor([True]*30 + [False]*20 + [True]*30)

        for profile in ['linear', 'sigmoid', 'step']:
            alpha = generate_morph_profile_silence_aware(80, is_speech, profile)

            assert torch.all(alpha >= 0.0), f"{profile}: Alpha contains values < 0"
            assert torch.all(alpha <= 1.0), f"{profile}: Alpha contains values > 1"


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
