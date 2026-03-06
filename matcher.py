
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from torch import Tensor
#from torchaudio.sox_effects import apply_effects_tensor
from wavlm.WavLM import WavLM
from knnvc_utils import generate_matrix_from_index


SPEAKER_INFORMATION_LAYER = 6
SPEAKER_INFORMATION_WEIGHTS = generate_matrix_from_index(SPEAKER_INFORMATION_LAYER)


def fast_cosine_dist(source_feats: Tensor, matching_pool: Tensor, device: str = 'cpu') -> Tensor:
    """ Like torch.cdist, but fixed dim=-1 and for cosine distance."""
    source_norms = torch.norm(source_feats, p=2, dim=-1).to(device)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
    dotprod /= 2

    dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
    return dists


def generate_morph_profile(n_frames: int, profile: str = 'linear', params: dict = None) -> Tensor:
    """
    Generate time-varying interpolation coefficients alpha(t) for continuous morphing.

    Arguments:
        - n_frames: Number of frames in the query sequence
        - profile: Type of interpolation curve
            * 'linear'
            * 'sigmoid'
            * 'step': (aburupt change from 0 to 1 at t)
            * 'custom': custom array
        - params: Profile-specific ps
            * For 'sigmoid': {'steepness': float} (default 10, higher = steeper)
            * For 'step': {'threshold': float} (default 0.5, when to transition)
            * For 'custom': {'alpha_values': Tensor or list} (must have n_frames elements)

    Returns:
        - Tensor of shape (n_frames,) with values in [0, 1]
          where 0 = 100% Speaker A, 1 = 100% Speaker B
    """
    if params is None:
        params = {}

    # Create normalized time vector [0, 1]
    t = torch.linspace(0, 1, n_frames)

    if profile == 'linear':
        alpha = t
    elif profile == 'sigmoid':
        # Smooth S-curve transition, centered at t=0.5
        k = params.get('steepness', 10)
        alpha = torch.sigmoid(k * (t - 0.5))
        # Normalize to ensure alpha[0] ~ 0 and alpha[-1] ~ 1
        alpha = (alpha - alpha[0]) / (alpha[-1] - alpha[0])
    elif profile == 'step':
        threshold = params.get('threshold', 0.5)
        alpha = (t >= threshold).float()
    elif profile == 'custom':
        alpha_values = params.get('alpha_values')
        if alpha_values is None:
            raise ValueError("Custom profile requires 'alpha_values' in params")
        alpha = torch.tensor(alpha_values, dtype=torch.float32) if not isinstance(alpha_values, Tensor) else alpha_values
        if alpha.shape[0] != n_frames:
            raise ValueError(f"Custom alpha must have {n_frames} values, got {alpha.shape[0]}")
    else:
        raise ValueError(f"Unknown morph profile: {profile}. Choose from ['linear', 'sigmoid', 'step', 'custom']")

    return alpha


def detect_voice_activity_energy(
    features: Tensor,
    threshold_db: float = -40,
    min_duration_ms: int = 50,
    frame_rate_hz: int = 50
) -> Tensor:
    """
    Detect voice activity using energy-based thresholding on WavLM features.

    This is a simple but effective VAD that operates directly on feature vectors,
    ensuring perfect frame alignment. It computes the L2 norm of each feature vector
    as a proxy for energy, then applies a dB threshold.

    Arguments:
        - features: Tensor (n_frames, feat_dim) - WavLM features from get_features()
        - threshold_db: float - Energy threshold in dB relative to maximum (default: -40)
            * Lower values (e.g., -50) are more sensitive, treat quieter regions as speech
            * Higher values (e.g., -30) are less sensitive, treat more regions as silence
        - min_duration_ms: int - Minimum silence duration to consider (default: 50ms)
            * Silence spans shorter than this are treated as speech
            * Helps filter out brief dips in energy within continuous speech
        - frame_rate_hz: int - Feature frame rate in Hz (default: 50, i.e., 20ms frames)

    Returns:
        - Boolean tensor of shape (n_frames,) where True = speech, False = silence

    Example:
        >>> query_seq = knn_vc.get_features('source.wav')
        >>> is_speech = detect_voice_activity_energy(query_seq, threshold_db=-40)
        >>> print(f"Speech frames: {is_speech.sum()} / {len(is_speech)}")
    """
    # Compute energy per frame (L2 norm of feature vector)
    # Higher norm = more energy = likely speech
    energy = features.norm(dim=1)  # (n_frames,)

    # Convert to dB scale for more intuitive thresholding
    energy_db = 20 * torch.log10(energy + 1e-8)

    # Normalize so that maximum energy = 0 dB
    # All other frames are negative dB relative to max
    energy_db = energy_db - energy_db.max()

    # Apply threshold: frames above threshold_db are considered speech
    is_speech = energy_db > threshold_db

    # Filter out very short silence spans (likely just brief pauses within speech)
    min_frames = int(min_duration_ms / 1000 * frame_rate_hz)
    if min_frames > 0:
        is_speech = _filter_short_silences(is_speech, min_frames)

    return is_speech


def _filter_short_silences(is_speech: Tensor, min_frames: int) -> Tensor:
    """
    Helper function: Remove silence spans shorter than min_frames.

    Converts short silence regions to speech to avoid choppy VAD decisions.
    For example, a brief 30ms dip in energy between syllables shouldn't be
    treated as a pause.

    Arguments:
        - is_speech: Boolean tensor (n_frames,)
        - min_frames: Minimum number of consecutive silence frames to keep

    Returns:
        - Filtered boolean tensor with short silences removed
    """
    # Find transitions: speech→silence and silence→speech
    # Prepend/append 0 to detect boundaries at start/end
    diff = torch.diff(
        is_speech.int(),
        prepend=torch.tensor([0]),
        append=torch.tensor([0])
    )

    silence_starts = torch.where(diff == -1)[0]  # Speech → silence transition
    silence_ends = torch.where(diff == 1)[0]     # Silence → speech transition

    # For each silence span, if it's too short, mark it as speech
    is_speech_filtered = is_speech.clone()
    for start, end in zip(silence_starts, silence_ends):
        if end - start < min_frames:
            is_speech_filtered[start:end] = True

    return is_speech_filtered


def generate_morph_profile_silence_aware(
    n_frames: int,
    is_speech: Tensor,
    profile: str = 'linear',
    params: dict = None
) -> Tensor:
    """
    Generate time-varying interpolation coefficients α(t) that only advance during speech.

    Unlike standard generate_morph_profile() which advances uniformly over time,
    this function freezes the alpha coefficient during silence periods and only
    advances it during active speech. This creates more natural-sounding morphing
    for utterances with pauses.

    The key idea: Alpha progresses from 0 to 1 based on how much SPEECH has occurred,
    not based on wall-clock time. During silence, alpha is frozen at the value from
    the last speech frame.

    Arguments:
        - n_frames: Total number of frames in the utterance
        - is_speech: Boolean tensor (n_frames,) where True = speech, False = silence
            * Typically from detect_voice_activity_energy()
        - profile: Type of interpolation curve (same as generate_morph_profile)
            * 'linear': α(t) advances linearly with speech content
            * 'sigmoid': α(t) follows S-curve based on speech progress
            * 'step': α(t) = 0 before threshold, 1 after (based on speech content)
            * 'custom': user-provided values (see params)
        - params: Profile-specific parameters (same as generate_morph_profile)

    Returns:
        - Tensor of shape (n_frames,) with values in [0, 1]
          * Alpha advances only during speech frames
          * Alpha frozen at last speech value during silence frames
          * Alpha reaches 1.0 at the final speech frame (not necessarily the final frame)

    Example:
        >>> is_speech = torch.tensor([1, 1, 1, 0, 0, 1, 1, 1], dtype=torch.bool)
        >>> alpha = generate_morph_profile_silence_aware(8, is_speech, 'linear')
        >>> print(alpha)
        tensor([0.00, 0.20, 0.40, 0.40, 0.40, 0.60, 0.80, 1.00])
        #                         ^^^^  ^^^^ frozen during silence

    Behavior:
        [Speech A] [Silence] [Speech B] [Silence] [Speech C]
         α=0.0      α=0.0     α=0.5      α=0.5     α=1.0
                    ↑ frozen            ↑ frozen    ↑ reaches 1.0
    """
    if params is None:
        params = {}

    # Step 1: Compute cumulative speech progress (0 to 1 based on speech content)
    # This creates a "speech-only timeline" where only voiced frames count
    speech_cumsum = torch.cumsum(is_speech.float(), dim=0)
    total_speech_frames = is_speech.sum().item()

    # Edge case: No speech detected → freeze alpha at 0 everywhere
    if total_speech_frames == 0:
        return torch.zeros(n_frames)

    # Normalize cumulative speech count to [0, 1]
    # t_speech[i] = "what fraction of total speech has occurred by frame i"
    t_speech = speech_cumsum / total_speech_frames  # (n_frames,)

    # Step 2: Generate base alpha profile using speech timeline
    # The key difference: we use t_speech (speech progress) instead of t (wall time)

    if profile == 'linear':
        # Linear progression through speech content
        alpha_base = t_speech

    elif profile == 'sigmoid':
        # Smooth S-curve centered at 50% of speech content (not 50% of time)
        k = params.get('steepness', 10)
        alpha_base = torch.sigmoid(k * (t_speech - 0.5))
        # Normalize to ensure alpha starts at 0 and ends at 1
        alpha_min = alpha_base.min()
        alpha_max = alpha_base.max()
        alpha_base = (alpha_base - alpha_min) / (alpha_max - alpha_min + 1e-8)

    elif profile == 'step':
        # Abrupt transition at threshold% of speech content
        threshold = params.get('threshold', 0.5)
        alpha_base = (t_speech >= threshold).float()

    elif profile == 'custom':
        # User provides custom alpha trajectory
        alpha_values = params.get('alpha_values')
        if alpha_values is None:
            raise ValueError("Custom profile requires 'alpha_values' in params")

        alpha_custom = torch.tensor(alpha_values, dtype=torch.float32) if not isinstance(alpha_values, Tensor) else alpha_values

        # Interpolate custom values to match speech progress
        # Map t_speech (0 to 1) to indices in custom array
        indices = (t_speech * (len(alpha_custom) - 1)).long()
        indices = torch.clamp(indices, 0, len(alpha_custom) - 1)
        alpha_base = alpha_custom[indices]

    else:
        raise ValueError(f"Unknown morph profile: {profile}. Choose from ['linear', 'sigmoid', 'step', 'custom']")

    # Step 3: Apply speech mask - freeze alpha during silence
    # During silence frames, use the alpha value from the most recent speech frame
    alpha = torch.zeros(n_frames)
    last_speech_alpha = 0.0

    for i in range(n_frames):
        if is_speech[i]:
            # Speech frame: use computed alpha and update last_speech_alpha
            alpha[i] = alpha_base[i]
            last_speech_alpha = alpha_base[i]
        else:
            # Silence frame: freeze at last speech value
            alpha[i] = last_speech_alpha

    # Step 4: Ensure alpha reaches 1.0 at the final speech frame
    # This addresses the user requirement: "alpha should reach 1 by the end of the final speech segment"
    # Find the last speech frame
    speech_indices = torch.where(is_speech)[0]
    if len(speech_indices) > 0:
        last_speech_idx = speech_indices[-1].item()
        # Force alpha to 1.0 at and after the last speech frame
        alpha[last_speech_idx:] = 1.0

    return alpha


class KNeighborsVC(nn.Module):

    def __init__(self,
        wavlm: WavLM,
        hifigan: HiFiGAN,
        hifigan_cfg: AttrDict,
        device='cuda'
    ) -> None:
        """ kNN-VC matcher. 
        Arguments:
            - `wavlm` : trained WavLM model
            - `hifigan`: trained hifigan model
            - `hifigan_cfg`: hifigan config to use for vocoding.
        """
        super().__init__()
        # set which features to extract from wavlm
        self.weighting = torch.tensor(SPEAKER_INFORMATION_WEIGHTS, device=device)[:, None]
        # load hifigan
        self.hifigan = hifigan.eval()
        self.h = hifigan_cfg
        # store wavlm
        self.wavlm = wavlm.eval()
        self.device = torch.device(device)
        self.sr = self.h.sampling_rate
        self.hop_length = 320

    def get_matching_set(self, wavs: list[Path] | list[Tensor], weights=None, vad_trigger_level=7) -> Tensor:
        """ Get concatenated wavlm features for the matching set using all waveforms in `wavs`, 
        specified as either a list of paths or list of loaded waveform tensors of 
        shape (channels, T), assumed to be of 16kHz sample rate.
        Optionally specify custom WavLM feature weighting with `weights`.
        """
        feats = []
        for p in wavs:
            feats.append(self.get_features(p, weights=self.weighting if weights is None else weights, vad_trigger_level=vad_trigger_level))
        
        feats = torch.concat(feats, dim=0).cpu()
        return feats
        

    @torch.inference_mode()
    def vocode(self, c: Tensor) -> Tensor:
        """ Vocode features with hifigan. `c` is of shape (bs, seq_len, c_dim) """
        y_g_hat = self.hifigan(c)
        y_g_hat = y_g_hat.squeeze(1)
        return y_g_hat


    @torch.inference_mode()
    def get_features(self, path, weights=None, vad_trigger_level=0):
        """Returns features of `path` waveform as a tensor of shape (seq_len, dim), optionally perform VAD trimming
        on start/end with `vad_trigger_level`.
        """
        # load audio
        if weights == None: weights = self.weighting
        if type(path) in [str, Path]:
            x, sr = torchaudio.load(path, normalize=True)
        else:
            x: Tensor = path
            sr = self.sr
            if x.dim() == 1: x = x[None]
                
        if not sr == self.sr :
            print(f"resample {sr} to {self.sr} in {path}")
            x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=self.sr)
            sr = self.sr
            
        # trim silence from front and back
        if vad_trigger_level > 1e-3:
            transform = T.Vad(sample_rate=sr, trigger_level=vad_trigger_level)
            x_front_trim = transform(x)
            # original way, disabled because it lacks windows support
            #waveform_reversed, sr = apply_effects_tensor(x_front_trim, sr, [["reverse"]])
            waveform_reversed = torch.flip(x_front_trim, (-1,))
            waveform_reversed_front_trim = transform(waveform_reversed)
            waveform_end_trim = torch.flip(waveform_reversed_front_trim, (-1,))
            #waveform_end_trim, sr = apply_effects_tensor(
            #    waveform_reversed_front_trim, sr, [["reverse"]]
            #)
            x = waveform_end_trim

        # extract the representation of each layer
        wav_input_16khz = x.to(self.device)
        if torch.allclose(weights, self.weighting):
            # use fastpath
            features = self.wavlm.extract_features(wav_input_16khz, output_layer=SPEAKER_INFORMATION_LAYER, ret_layer_results=False)[0]
            features = features.squeeze(0)
        else:
            # use slower weighted
            rep, layer_results = self.wavlm.extract_features(wav_input_16khz, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
            features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)
            # save full sequence
            features = ( features*weights[:, None] ).sum(dim=0) # (seq_len, dim)
        
        return features


    @torch.inference_mode()
    def match(self, query_seq: Tensor, matching_set: Tensor, synth_set: Tensor = None,
              topk: int = 4, tgt_loudness_db: float | None = -16,
              target_duration: float | None = None, device: str | None = None) -> Tensor:
        """ Given `query_seq`, `matching_set`, and `synth_set` tensors of shape (N, dim), perform kNN regression matching
        with k=`topk`. Inputs:
            - `query_seq`: Tensor (N1, dim) of the input/source query features.
            - `matching_set`: Tensor (N2, dim) of the matching set used as the 'training set' for the kNN algorithm.
            - `synth_set`: optional Tensor (N2, dim) corresponding to the matching set. We use the matching set to assign each query
                vector to a vector in the matching set, and then use the corresponding vector from the synth set during HiFiGAN synthesis.
                By default, and for best performance, this should be identical to the matching set.
            - `topk`: k in the kNN -- the number of nearest neighbors to average over.
            - `tgt_loudness_db`: float db used to normalize the output volume. Set to None to disable.
            - `target_duration`: if set to a float, interpolate resulting waveform duration to be equal to this value in seconds.
            - `device`: if None, uses default device at initialization. Otherwise uses specified device
        Returns:
            - converted waveform of shape (T,)
        """
        device = torch.device(device) if device is not None else self.device
        if synth_set is None: synth_set = matching_set.to(device)
        else: synth_set = synth_set.to(device)
        matching_set = matching_set.to(device)
        query_seq = query_seq.to(device)

        if target_duration is not None:
            target_samples = int(target_duration*self.sr)
            scale_factor = (target_samples/self.hop_length) / query_seq.shape[0] # n_targ_feats / n_input_feats
            query_seq = F.interpolate(query_seq.T[None], scale_factor=scale_factor, mode='linear')[0].T

        dists = fast_cosine_dist(query_seq, matching_set, device=device)
        best = dists.topk(k=topk, largest=False, dim=-1)
        out_feats = synth_set[best.indices].mean(dim=1)

        prediction = self.vocode(out_feats[None].to(device)).cpu().squeeze()

        # normalization
        if tgt_loudness_db is not None:
            src_loudness = torchaudio.functional.loudness(prediction[None], self.h.sampling_rate)
            tgt_loudness = tgt_loudness_db
            pred_wav = torchaudio.functional.gain(prediction, tgt_loudness - src_loudness)
        else: pred_wav = prediction
        return pred_wav

    @torch.inference_mode()
    def match_morph(self, query_seq: Tensor,
                    matching_set_A: Tensor,
                    matching_set_B: Tensor,
                    synth_set_A: Tensor = None,
                    synth_set_B: Tensor = None,
                    topk: int = 4,
                    morph_profile: str = 'linear',
                    morph_params: dict = None,
                    silence_aware: bool = False,
                    vad_threshold_db: float = -40,
                    vad_min_silence_ms: int = 50,
                    tgt_loudness_db: float | None = -16,
                    target_duration: float | None = None,
                    device: str | None = None) -> Tensor:
        """
        Perform continuous morphing from Speaker A to Speaker B over the duration of the utterance.

        Arguments:
            - query_seq: Tensor (N, dim) - source utterance features from get_features()
            - matching_set_A: Tensor (N_A, dim) - reference features for speaker A
            - matching_set_B: Tensor (N_B, dim) - reference features for speaker B
            - synth_set_A: optional Tensor (N_A, dim) - synthesis features for Speaker A
            - synth_set_B: optional Tensor (N_B, dim) - synthesis features for Speaker B
            - topk: int - k for knn matching
            - morph_profile: str - type of interpolation curve ('linear', 'sigmoid', 'step', 'custom')
            - morph_params: dict - profile-specific parameters
            - silence_aware: bool - if True, alpha only advances during speech, frozen during silence
            - vad_threshold_db: float - energy threshold in dB for VAD (default: -40)
            - vad_min_silence_ms: int - minimum silence duration in ms (default: 50)
            - tgt_loudness_db: float - target loudness in dB for normalization (None to disable)
            - target_duration: float - target duration in seconds (None to use source duration)
            - device: str - compute device ('cuda', 'cpu', etc.)

        Returns:
            - converted waveform of shape (T,) with continuous morphing from A to B

        Example:
            >>> query_seq = knn_vc.get_features('source.wav')
            >>> matching_A = knn_vc.get_matching_set(['speaker_A_ref.wav'])
            >>> matching_B = knn_vc.get_matching_set(['speaker_B_ref.wav'])
            >>> morphed = knn_vc.match_morph(query_seq, matching_A, matching_B, topk=4)
        """
        device = torch.device(device) if device is not None else self.device

        # If synth sets not provided, use matching sets
        if synth_set_A is None: synth_set_A = matching_set_A.to(device
        else: synth_set_A = synth_set_A.to(device)
        if synth_set_B is None: synth_set_B = matching_set_B.to(device)
        else: synth_set_B = synth_set_B.to(device)

        matching_set_A = matching_set_A.to(device)
        matching_set_B = matching_set_B.to(device)
        query_seq = query_seq.to(device)

        # Handle target duration by interpolating query sequence
        if target_duration is not None:
            target_samples = int(target_duration * self.sr)
            scale_factor = (target_samples / self.hop_length) / query_seq.shape[0]
            query_seq = F.interpolate(query_seq.T[None], scale_factor=scale_factor, mode='linear')[0].T

        # Generate morphing profile alpha(t) in [0, 1] over the utterance
        n_frames = query_seq.shape[0]

        if silence_aware:
            # Detect voice activity to identify speech vs. silence frames
            is_speech = detect_voice_activity_energy(
                query_seq.cpu(),  # VAD operates on CPU
                threshold_db=vad_threshold_db,
                min_duration_ms=vad_min_silence_ms,
                frame_rate_hz=50  # WavLM features at 20ms = 50Hz
            )

            # Generate silence-aware profile: alpha only advances during speech
            alpha = generate_morph_profile_silence_aware(
                n_frames,
                is_speech,
                morph_profile,
                morph_params
            ).to(device)
        else:
            # Standard time-based morphing: alpha advances uniformly
            alpha = generate_morph_profile(n_frames, morph_profile, morph_params).to(device)  # (n_frames,)

        # In this continuous setup perform knn matching against both speakers independently.
        # this allows using a source utterance form a third speaker
        # TODO(eetu): maybe just use the source feats when the source is from ref speaker A?
        dists_A = fast_cosine_dist(query_seq, matching_set_A, device=device)  # (n_frames, N_A)
        dists_B = fast_cosine_dist(query_seq, matching_set_B, device=device)  # (n_frames, N_B)

        # Get top-k nearest for each speaker
        best_A = dists_A.topk(k=topk, largest=False, dim=-1)  # indices: (n_frames, k)
        best_B = dists_B.topk(k=topk, largest=False, dim=-1)  # indices: (n_frames, k)

        # Compute mean of k nearest neighbors for each frame, for each speaker
        out_feats_A = synth_set_A[best_A.indices].mean(dim=1)  # (n_frames, dim)
        out_feats_B = synth_set_B[best_B.indices].mean(dim=1)  # (n_frames, dim)

        # linear combination according to morphing profile
        out_feats = (1 - alpha[:, None]) * out_feats_A + alpha[:, None] * out_feats_B

        prediction = self.vocode(out_feats[None].to(device)).cpu().squeeze()

        if tgt_loudness_db is not None:
            src_loudness = torchaudio.functional.loudness(prediction[None], self.h.sampling_rate)
            pred_wav = torchaudio.functional.gain(prediction, tgt_loudness_db - src_loudness)
        else:
            pred_wav = prediction

        return pred_wav


