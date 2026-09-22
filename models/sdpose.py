"""SDPose-Wholebody as the pose model: 133 COCO-WholeBody keypoints from a Stable
Diffusion 2 U-Net used as a feature extractor.

SDPose (arXiv 2509.24980, MIT, https://github.com/T-S-Liang/SDPose-OOD) does not denoise.
The person crop is encoded by the SD VAE, the U-Net is called **once** at a fixed timestep
with an empty prompt and a fixed task embedding, and the 640-channel feature of the last
up block - the one the reference picks for the 133-keypoint model (`models/ModifiedUNet.py`
takes `feats[::-1][1]`) - goes through an mmpose HeatmapHead whose UDP codec decodes the
keypoints. The scheduler is in x0 ("sample") mode, so there is no loop to run: the
reference's `scripts/eval.sh` and its gradio demo both pass the single timestep 999.

The weights are Comfy-Org's repackaging of that reference checkpoint into ComfyUI's own
format, so the U-Net, the VAE and the head all load through
`comfy.sd.load_checkpoint_guess_config` and ComfyUI offloads them like every other model.
The reference publishes 5.2 GB of fp32 diffusers weights; 1.4 GB of that is a CLIP text
encoder that only ever encodes the empty prompt, which ComfyUI inlines as a constant
(`comfy_extras/nodes_lotus.LotusConditioning`), hence the 1.9 GB here.
"""
import os

import numpy as np
import torch

import comfy.sample
import comfy.samplers
import comfy.sd
import folder_paths

from .. import log
from ..pose_utils.pose2d_utils import transform_preds
from .download import download

DEFAULT_SDPOSE = "sdpose_wholebody_fp16.safetensors"
DEFAULT_SDPOSE_URL = "https://huggingface.co/Comfy-Org/SDPose/resolve/main/checkpoints/sdpose_wholebody_fp16.safetensors"

# The fixed diffusion timestep the reference runs the U-Net at. Nothing here picks it
# directly - one `simple` step starts at the largest sigma, which is this timestep - so it
# is checked when the model is built rather than assumed.
TIMESTEP = 999
# The 133-keypoint head reads a 640-channel up-block feature; the last output block at
# that width is the one the reference selects.
FEATURE_CHANNELS = 640
# The heatmap is a quarter of the input on each axis (mmpose UDPHeatmap, sigma 6).
HEATMAP_SCALE = 4


class SDPose:
    """SDPose-Wholebody, called like the pose model it replaces: person crops in,
    keypoints in frame coordinates out. `patchers` are its two ComfyUI models, the U-Net
    and the VAE, so ComfyUI loads and offloads them with the detector - and together, or
    the VAE's own load would push the U-Net out once per frame."""

    def __init__(self, checkpoint):
        model, _, vae, _ = comfy.sd.load_checkpoint_guess_config(checkpoint, output_vae=True, output_clip=False)
        head = getattr(model.model.diffusion_model, "heatmap_head", None)
        if head is None:
            raise ValueError(f"{os.path.basename(checkpoint)} carries no SDPose heatmap head; the pose model is "
                             f"the SDPose checkpoint from {DEFAULT_SDPOSE_URL}")
        sampling = model.model.model_sampling
        sigmas = comfy.samplers.calculate_sigmas(sampling, "simple", 1)
        timestep = int(sampling.timestep(sigmas[:1])[0])
        if timestep != TIMESTEP:
            raise RuntimeError(f"one sampling step of {os.path.basename(checkpoint)} lands on timestep {timestep}, "
                               f"not the {TIMESTEP} SDPose is run at")
        self.patchers = [model, vae.patcher]
        self.vae = vae
        self.head = head
        heatmap_w, heatmap_h = (int(v) for v in head.heatmap_size)
        # (H, W) of the crop the head was trained on, the way `crop` takes its size
        self.input_resolution = (heatmap_h * HEATMAP_SCALE, heatmap_w * HEATMAP_SCALE)
        # the feature is read off the U-Net while it runs, from a clone so the patch does
        # not follow this model everywhere else in ComfyUI
        self.sampler_model = model.clone()
        self.sampler_model.model_options["transformer_options"] = {"patches": {"output_block_patch": [self._capture]}}
        self._feature = None
        self._conditioning = None
        log.info(f"SDPose: {self.input_resolution[0]}x{self.input_resolution[1]} crops, "
                 f"{head.final_layer.out_channels} keypoints, one U-Net call at timestep {timestep}")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _capture(self, h, hsp, transformer_options):
        """The output_block_patch ComfyUI calls before each output block, where `h` is the
        previous block's output. The last 640-channel one is the feature the head reads."""
        if h.shape[1] == FEATURE_CHANNELS:
            self._feature = h
        return h, hsp

    @property
    def conditioning(self):
        """The empty-prompt embedding of SD 2's text encoder, which is all the reference
        ever conditions on; ComfyUI ships it as a constant so the text encoder is not
        needed. The task embedding the reference builds from [1, 0] comes with the model
        (`comfy.model_base.Lotus.extra_conds`)."""
        if self._conditioning is None:
            from comfy_extras.nodes_lotus import LotusConditioning
            self._conditioning = LotusConditioning().execute().result[0]
        return self._conditioning

    def feature(self, latent):
        """The U-Net feature for a batch of latents: one call at TIMESTEP, x0 mode, no
        denoising loop. The sampler is only the way to get ComfyUI to run the U-Net with
        the conditioning the model asks for; its output is thrown away."""
        self._feature = None
        comfy.sample.sample(self.sampler_model, noise=torch.zeros_like(latent), steps=1, cfg=1.0,
                            sampler_name="euler", scheduler="simple",
                            positive=self.conditioning, negative=self.conditioning,
                            latent_image=latent, disable_noise=True, disable_pbar=True)
        if self._feature is None:
            raise RuntimeError("the SDPose U-Net produced no 640-channel feature; this checkpoint is not SDPose")
        feature, self._feature = self._feature, None
        return feature

    def forward(self, images, center, scale, **kwargs):
        """[N, 133, 3] keypoints as x, y, confidence in frame coordinates.

        `images` are the person crops [N, H, W, 3] in 0..1 at `input_resolution`, taken
        around `center` at `scale` - the same (centre, scale / 200) pair
        `bbox_from_detector` returns and `crop` cut them with."""
        images = torch.as_tensor(np.ascontiguousarray(images), dtype=torch.float32)
        if tuple(images.shape[1:3]) != self.input_resolution:
            raise ValueError(f"SDPose takes {self.input_resolution[0]}x{self.input_resolution[1]} crops, got "
                             f"{images.shape[1]}x{images.shape[2]}")
        with torch.no_grad():
            latent = self.vae.encode(images)
            keypoints, scores = self.head(self.feature(latent))
        # the head returns the keypoints on the input grid, [0, W - 1] x [0, H - 1]; the
        # UDP inverse of the crop puts them back on the frame
        size = [self.input_resolution[1], self.input_resolution[0]]
        return np.stack([
            np.concatenate([transform_preds(kps, c, np.asarray(s) * 200, size, use_udp=True), conf[:, None]], axis=1)
            for kps, conf, c, s in zip(keypoints, scores, center, scale)
        ]).astype(np.float32)


_loaded = {"name": None, "model": None}


def load_sdpose(name=DEFAULT_SDPOSE):
    """The SDPose checkpoint `name` as an SDPose model, built once and kept; the default
    checkpoint is downloaded into models/checkpoints when it is missing."""
    if _loaded["name"] == name:
        return _loaded["model"]
    path = folder_paths.get_full_path("checkpoints", name)
    if path is None and name == DEFAULT_SDPOSE:
        path = os.path.join(folder_paths.get_folder_paths("checkpoints")[0], name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        download(DEFAULT_SDPOSE_URL, path)
    if path is None:
        raise FileNotFoundError(f"SDPose checkpoint {name} is not in models/checkpoints")
    _loaded["name"], _loaded["model"] = None, None
    with log.step(f"building {name}"):
        model = SDPose(path)
    _loaded["name"], _loaded["model"] = name, model
    return model
