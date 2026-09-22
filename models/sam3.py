"""Person segmentation with SAM 3 / SAM 3.1, driven by the pose detector.

There is no text prompt and no tracker. Every frame is segmented from what the pose
pipeline already knows about it: the detector's person box and the body keypoints, given
to the SAM decoder as a box and positive points. Frames are independent, so nothing
drifts or ghosts over a long clip; each mask is as good as that frame's detection.

The model is ComfyUI's own SAM3 implementation loaded from `models/checkpoints`
(`sam3.1_multiplex_fp16.safetensors`, fetched on first use when missing), so it is loaded
and offloaded by ComfyUI like every other model.
"""
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import comfy.sd
import folder_paths
from comfy import model_management as mm
from comfy.utils import ProgressBar, common_upscale

from .. import log
from .download import download

DEFAULT_SAM3 = "sam3.1_multiplex_fp16.safetensors"
DEFAULT_SAM3_URL = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"

SAM3_SIZE = 1008
# Body keypoints used as positive points (nose, neck, shoulders, hips, knees in the
# node's body layout) - the same set the node exports as key_frame_body_points.
PROMPT_KEYPOINTS = (0, 1, 2, 5, 8, 11, 10, 13)
MIN_KEYPOINT_CONF = 0.3
# Islands smaller than this fraction of the largest region are decoder noise (specks in
# shadows and edges), not the person; downstream block masks would blow them up.
MIN_ISLAND_FRACTION = 0.01


def drop_islands(mask):
    """Remove connected regions of a binary uint8 mask far smaller than its largest one."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= areas.max() * MIN_ISLAND_FRACTION) + 1
    return np.isin(labels, keep).astype(np.uint8)


def sam3_choices():
    """SAM3 checkpoints in models/checkpoints (by name), plus the default before it exists."""
    names = [n for n in folder_paths.get_filename_list("checkpoints") if "sam3" in os.path.basename(n).lower()]
    return ["none"] + sorted(set(names) | {DEFAULT_SAM3})


_loaded = {"name": None, "model": None}


def load_sam3(name):
    """The SAM3 checkpoint `name` as a ComfyUI model, loaded once and kept; the default
    checkpoint is downloaded into models/checkpoints when it is missing."""
    if _loaded["name"] == name:
        return _loaded["model"]
    path = folder_paths.get_full_path("checkpoints", name)
    if path is None and name == DEFAULT_SAM3:
        path = os.path.join(folder_paths.get_folder_paths("checkpoints")[0], name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        download(DEFAULT_SAM3_URL, path)
    if path is None:
        raise FileNotFoundError(f"SAM3 checkpoint {name} is not in models/checkpoints")
    _loaded["name"], _loaded["model"] = None, None
    with log.step(f"loading {name}"):
        model = comfy.sd.load_checkpoint_guess_config(path, output_vae=False, output_clip=False)[0]
    _loaded["name"], _loaded["model"] = name, model
    return model


def segment_frames(model, images, bboxes, pose_metas, refine=True):
    """[N, H, W] float masks of the detected person in `images` [N, H, W, 3].

    `bboxes[i]` is frame i's detector box as (x1, y1, x2, y2, score) or None;
    `pose_metas[i]["keypoints_body"]` its body keypoints, normalised to the frame, with
    confidence. A frame with neither a box nor a confident keypoint gets an empty mask."""
    N, H, W, _ = images.shape
    mm.load_model_gpu(model)
    device, dtype = mm.get_torch_device(), model.model.get_dtype()
    sam3 = model.model.diffusion_model
    sx, sy = SAM3_SIZE / W, SAM3_SIZE / H
    masks = torch.zeros(N, H, W)
    pbar = ProgressBar(N)
    for i in range(N):
        box_inputs = point_inputs = None
        bbox = bboxes[i]
        if bbox is not None and bbox[-1] > 0:
            box_inputs = torch.tensor([[[bbox[0] * sx, bbox[1] * sy], [bbox[2] * sx, bbox[3] * sy]]],
                                      device=device, dtype=dtype)
        kps = pose_metas[i]["keypoints_body"]
        points = [(kps[k][0] * SAM3_SIZE, kps[k][1] * SAM3_SIZE) for k in PROMPT_KEYPOINTS if kps[k][2] > MIN_KEYPOINT_CONF]
        if points:
            point_inputs = {"point_coords": torch.tensor([points], device=device, dtype=dtype),
                            "point_labels": torch.ones(1, len(points), dtype=torch.int32, device=device)}
        if box_inputs is None and point_inputs is None:
            pbar.update(1)
            continue
        frame = common_upscale(images[i:i + 1, ..., :3].movedim(-1, 1), SAM3_SIZE, SAM3_SIZE, "bilinear", crop="disabled")
        frame = frame.to(device=device, dtype=dtype)
        with torch.inference_mode():
            logits = sam3.forward_segment(frame, point_inputs=point_inputs, box_inputs=box_inputs)
            if refine:
                logits = sam3.forward_segment(frame, mask_inputs=logits)
            mask = F.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        masks[i] = torch.from_numpy(drop_islands((mask > 0).cpu().numpy().astype(np.uint8))).float()
        pbar.update(1)
    return masks
