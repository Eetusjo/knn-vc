dependencies = ['torch', 'torchaudio', 'numpy']

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import logging
import json
from pathlib import Path


from wavlm.WavLM import WavLM, WavLMConfig
from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from matcher import KNeighborsVC
from bigvgan_vocoder import (
    BigVGANVocoder,
    BigVGANDirectVocoder,
    build_projection,
    load_bigvgan_partial,
)


def knn_vc(pretrained=True, progress=True, prematched=True, device='cuda',
           vocoder='hifigan') -> KNeighborsVC:
    """ Load kNN-VC (WavLM encoder and vocoder decoder).
    Arguments:
        - pretrained: load pretrained weights
        - progress: show download progress
        - prematched: use prematched HiFiGAN weights (only applies to hifigan vocoder)
        - device: compute device
        - vocoder: 'hifigan' (default), 'bigvgan' (projection path),
                   or 'bigvgan_direct' (no projection — WavLM features fed
                   straight to BigVGAN with hop=480, num_mels=1024).
    """
    wavlm = wavlm_large(pretrained, progress, device)

    if vocoder == 'bigvgan':
        hifigan, hifigan_cfg = bigvgan_wavlm(pretrained, device)
    elif vocoder == 'bigvgan_direct':
        hifigan, hifigan_cfg = bigvgan_wavlm_direct(pretrained, device)
    else:
        hifigan, hifigan_cfg = hifigan_wavlm(pretrained, progress, prematched, device)

    knnvc = KNeighborsVC(wavlm, hifigan, hifigan_cfg, device)
    return knnvc


def hifigan_wavlm(pretrained=True, progress=True, prematched=True, device='cuda') -> HiFiGAN:
    """ Load pretrained hifigan trained to vocode wavlm features. Optionally use weights trained on `prematched` data. """
    cp = Path(__file__).parent.absolute()

    with open(cp/'hifigan'/'config_v1_wavlm.json') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    device = torch.device(device)

    generator = HiFiGAN(h).to(device)
    
    if pretrained:
        if prematched:
            url = "https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt"
        else:
            url = "https://github.com/bshall/knn-vc/releases/download/v0.1/g_02500000.pt"
        state_dict_g = torch.hub.load_state_dict_from_url(
            url,
            map_location=device,
            progress=progress
        )
        generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    print(f"[HiFiGAN] Generator loaded with {sum([p.numel() for p in generator.parameters()]):,d} parameters.")
    return generator, h


def bigvgan_wavlm(pretrained=True, device='cuda', checkpoint_path=None):
    """ Load BigVGAN v2 vocoder adapted for WavLM features.
    Arguments:
        - pretrained: load pretrained BigVGAN weights from HuggingFace
        - device: compute device
        - checkpoint_path: optional path to fine-tuned checkpoint (.pt file with
          'projection' key and optionally 'bigvgan' key)
    Returns:
        - (BigVGANVocoder, AttrDict config) matching the hifigan_wavlm() interface
    """
    import bigvgan as bigvgan_module

    device = torch.device(device)

    # Load pretrained BigVGAN from HuggingFace.
    # Note: huggingface_hub>=1.0 removed 'proxies'/'resume_download' from the
    # from_pretrained() → _from_pretrained() call chain, but bigvgan 2.x still
    # declares them as required keyword args in _from_pretrained. Call directly.
    model = bigvgan_module.BigVGAN._from_pretrained(
        model_id='nvidia/bigvgan_v2_24khz_100band_256x',
        revision=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=False,
        local_files_only=False,
        token=None,
        use_cuda_kernel=False,
        map_location=str(device),
    )

    # Load fine-tuned weights if a checkpoint is provided
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        ckpt_mode = ckpt.get('mode', 'projection')
        if ckpt_mode != 'projection':
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' was trained in mode={ckpt_mode!r}; "
                f"use bigvgan_wavlm_direct() to load it.")
        proj_type = ckpt.get('projection_type', 'linear')
        projection = build_projection(proj_type, in_dim=1024, out_dim=model.h.num_mels)
        projection.load_state_dict(ckpt['projection'])
        if 'bigvgan' in ckpt:
            model.load_state_dict(ckpt['bigvgan'])
        print(f"[BigVGAN] Loaded fine-tuned checkpoint from {checkpoint_path} (projection: {proj_type})")
    else:
        projection = build_projection('deconv', in_dim=1024, out_dim=model.h.num_mels)
        print("[BigVGAN] Using pretrained BigVGAN with a randomly initialized 'deconv' projection.")
        print("[BigVGAN] Note: fine-tune with train_bigvgan.py for good quality.")

    vocoder = BigVGANVocoder(model, projection, target_sr=model.h.sampling_rate).to(device)
    vocoder.eval()
    vocoder.remove_weight_norm()
    print(f"[BigVGAN] Loaded with {sum(p.numel() for p in vocoder.parameters()):,d} parameters "
          f"({sum(p.numel() for p in vocoder.projection.parameters()):,d} in projection layer).")

    # Build a config AttrDict compatible with KNeighborsVC expectations.
    # Output stays at BigVGAN's native rate (24 kHz) — no downsampling to 16 kHz.
    cfg = AttrDict({
        'sampling_rate': model.h.sampling_rate,
        'num_mels': model.h.num_mels,
        'hop_size': model.h.hop_size,
    })

    return vocoder, cfg


def bigvgan_wavlm_direct(pretrained=True, device='cuda', checkpoint_path=None,
                          config_path=None):
    """ Load BigVGAN configured to consume WavLM features directly.

    Differs from `bigvgan_wavlm` in that there is no projection layer or
    frame-rate adapter: BigVGAN's `conv_pre` is widened to 1024 input channels
    and `hop_size` is set to 480 so that 50 Hz WavLM frames produce 24 kHz
    audio at the correct rate end-to-end.

    Arguments:
        - pretrained: if True and `checkpoint_path` is None, partial-load
          weights from NVIDIA's pretrained 100-band 24 kHz BigVGAN-v2
          (everything except conv_pre and the changed upsamplers transfers).
        - device: compute device
        - checkpoint_path: optional path to fine-tuned checkpoint (.pt file
          with a 'bigvgan' key holding the direct-mode state_dict).
        - config_path: optional override for the BigVGAN config JSON. Defaults
          to BigVGAN/configs/bigvgan_v2_24khz_wavlm_480x.json.

    Returns:
        - (BigVGANDirectVocoder, AttrDict config) matching the hifigan_wavlm()
          interface.
    """
    import bigvgan as bigvgan_module
    from bigvgan.env import AttrDict as BVGAttrDict

    device = torch.device(device)

    cp = Path(__file__).parent.absolute()
    if config_path is None:
        config_path = cp / 'BigVGAN' / 'configs' / 'bigvgan_v2_24khz_wavlm_480x.json'
    with open(config_path) as f:
        h = BVGAttrDict(json.loads(f.read()))

    model = bigvgan_module.BigVGAN(h).to(device)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        if ckpt.get('mode', 'projection') != 'direct':
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' was trained in "
                f"mode={ckpt.get('mode', 'projection')!r}, expected 'direct'.")
        model.load_state_dict(ckpt['bigvgan'])
        print(f"[BigVGAN-direct] Loaded fine-tuned checkpoint from {checkpoint_path}.")
    elif pretrained:
        # Warm-start from the 100-band 24 kHz pretrained BigVGAN. conv_pre and
        # the three upsamplers whose strides changed get random init; the AMP
        # resblocks (channel counts unchanged), conv_post, and the unchanged
        # upsamplers transfer.
        pretrained_model = bigvgan_module.BigVGAN._from_pretrained(
            model_id='nvidia/bigvgan_v2_24khz_100band_256x',
            revision=None,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            local_files_only=False,
            token=None,
            use_cuda_kernel=False,
            map_location=str(device),
        )
        load_bigvgan_partial(model, pretrained_model.state_dict())
        del pretrained_model
        print("[BigVGAN-direct] Warm-started from nvidia/bigvgan_v2_24khz_100band_256x; "
              "fine-tune with train_bigvgan.py --mode direct for good quality.")
    else:
        print("[BigVGAN-direct] Using random-init BigVGAN. Train before use.")

    vocoder = BigVGANDirectVocoder(model, target_sr=h.sampling_rate).to(device)
    vocoder.eval()
    vocoder.remove_weight_norm()
    print(f"[BigVGAN-direct] Loaded with {sum(p.numel() for p in vocoder.parameters()):,d} parameters.")

    cfg = AttrDict({
        'sampling_rate': h.sampling_rate,
        'num_mels': h.num_mels,    # 1024 — input dim, not actual mels
        'hop_size': h.hop_size,    # 480
    })
    return vocoder, cfg


def wavlm_large(pretrained=True, progress=True, device='cuda') -> WavLM:
    """Load the WavLM large checkpoint from the original paper. See https://github.com/microsoft/unilm/tree/master/wavlm for details. """
    if torch.cuda.is_available() == False:
        if str(device) != 'cpu':
            logging.warning(f"Overriding device {device} to cpu since no GPU is available.")
            device = 'cpu'
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt", 
        map_location=device, 
        progress=progress
    )
    
    cfg = WavLMConfig(checkpoint['cfg'])
    device = torch.device(device)
    model = WavLM(cfg)
    if pretrained:
        model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print(f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")
    return model
