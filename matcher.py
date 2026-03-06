
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
        # Normalize to ensure alpha[0] ≈ 0 and alpha[-1] ≈ 1
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


