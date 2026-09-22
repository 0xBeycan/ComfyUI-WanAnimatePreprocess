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
from comfy.ops import cast_to_input
from comfy.utils import ProgressBar, common_upscale

from .. import log
from .download import download

DEFAULT_SAM3 = "sam3.1_multiplex_fp16.safetensors"
DEFAULT_SAM3_URL = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"

SAM3_SIZE = 1008
# Body keypoints used as positive points: nose, neck, both shoulders, both hips and both
# ankles in the body layout (see guard.BODY_NAMES).
PROMPT_KEYPOINTS = (0, 1, 2, 5, 8, 11, 10, 13)
NOSE, NECK, R_SHOULDER, L_SHOULDER, R_HIP, L_HIP = 0, 1, 2, 5, 8, 11
# Keypoints sit on joints, so dark low-texture clothing between them carries no positive
# evidence and the decoder drops parts of it (a close-up lost the lower half of a jacket).
# These fractions down the shoulder-to-hip line, or below the shoulders when the hips are
# out of frame, put points on the clothing itself.
TORSO_FRACTIONS = (0.35, 0.65)
MIN_KEYPOINT_CONF = 0.3
# The previous frame's mask is given to the decoder as a prior alongside the box and the
# points: on a close-up, where the person fills the frame and the boundary is ambiguous,
# an independent decision per frame makes the mask flicker. The prior is dropped when the
# detector box jumps (a cut, or another subject), so a wrong mask cannot carry on.
MIN_PRIOR_BOX_IOU = 0.5
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


def fill_holes(mask):
    """Fill background regions the mask fully encloses: a person has no holes, so a hole is
    clothing or hair the decoder dropped."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask == 0).astype(np.uint8), connectivity=4)
    if count <= 2:
        return mask
    H, W = mask.shape
    out = mask.copy()
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if x > 0 and y > 0 and x + w < W and y + h < H:   # does not touch the frame border
            out[labels == i] = 1
    return out


def torso_points(kps, threshold):
    """Points on the clothing between the shoulders and the hips, in normalised coordinates."""
    if kps[R_SHOULDER][2] <= threshold or kps[L_SHOULDER][2] <= threshold:
        return []
    shoulder = ((kps[R_SHOULDER][0] + kps[L_SHOULDER][0]) / 2, (kps[R_SHOULDER][1] + kps[L_SHOULDER][1]) / 2)
    hips = [kps[i] for i in (R_HIP, L_HIP) if kps[i][2] > threshold]
    if hips:
        hip = (sum(h[0] for h in hips) / len(hips), sum(h[1] for h in hips) / len(hips))
        return [(shoulder[0] + t * (hip[0] - shoulder[0]), shoulder[1] + t * (hip[1] - shoulder[1])) for t in TORSO_FRACTIONS]
    # the hips are out of frame: step down from the shoulders by their own width
    span = abs(kps[R_SHOULDER][0] - kps[L_SHOULDER][0]) or 0.15
    return [(shoulder[0], min(shoulder[1] + t * span, 0.99)) for t in (1.0, 2.0)]


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


def has_fast_path(sam3):
    """Whether this ComfyUI's SAM3 exposes the pieces `decode` needs to run the image
    encoder once per frame; decided once per model, and logged, so a core change falls
    back to the public two-pass path rather than failing mid-run."""
    if not hasattr(sam3, "_wanpre_fast_path"):
        bb = getattr(getattr(sam3, "detector", None), "backbone", {})
        ok = (isinstance(bb, (dict, torch.nn.ModuleDict)) and "vision_backbone" in bb
              and hasattr(bb["vision_backbone"], "multiplex") and hasattr(sam3.detector, "scalp")
              and hasattr(sam3, "tracker") and hasattr(sam3.tracker, "_forward_sam_heads"))
        if not ok:
            log.warning("this ComfyUI's SAM3 internals differ; running the image encoder twice per frame")
        sam3._wanpre_fast_path = ok
    return sam3._wanpre_fast_path


def decode(sam3, frame, point_inputs, box_inputs, refine, mask_prior=None):
    """Mask logits for one 1008x1008 frame from box / point prompts and an optional mask
    prior, with an optional refinement pass that feeds the first mask back to the decoder.
    This is SAM3Model.forward_segment with the image encoder run once: its features do not
    depend on the prompt, so the later passes only run the SAM heads."""
    if not has_fast_path(sam3):
        logits = sam3.forward_segment(frame, point_inputs=point_inputs, box_inputs=box_inputs, mask_inputs=mask_prior)
        return sam3.forward_segment(frame, mask_inputs=logits) if refine else logits
    bb = sam3.detector.backbone["vision_backbone"]
    if bb.multiplex:
        _, _, feats, _ = bb(frame, tracker_mode="interactive")
    else:
        _, _, feats, _ = bb(frame, need_tracker=True)
        if sam3.detector.scalp > 0:
            feats = feats[:-sam3.detector.scalp]
    high_res, backbone_feat = list(feats[:-1]), feats[-1]
    tracker = sam3.tracker
    no_mem = getattr(tracker, "interactivity_no_mem_embed", None)
    if no_mem is None:
        no_mem = getattr(tracker, "no_mem_embed", None)
    if no_mem is not None:
        B, C, H, W = backbone_feat.shape
        flat = backbone_feat.flatten(2).permute(0, 2, 1)
        backbone_feat = (flat + cast_to_input(no_mem, flat)).view(B, H, W, C).permute(0, 3, 1, 2)
    num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
    _, logits, _, _ = tracker._forward_sam_heads(
        backbone_features=backbone_feat, point_inputs=point_inputs, mask_inputs=mask_prior, box_inputs=box_inputs,
        high_res_features=high_res, multimask_output=(0 < num_pts <= 1 and mask_prior is None))
    if refine:
        _, logits, _, _ = tracker._forward_sam_heads(
            backbone_features=backbone_feat, point_inputs=None, mask_inputs=logits, box_inputs=None,
            high_res_features=high_res, multimask_output=False)
    return logits


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def segment_frames(model, images, bboxes, pose_metas, refine=True, temporal=True):
    """[N, H, W] float masks of the detected person in `images` [N, H, W, 3].

    `bboxes[i]` is frame i's detector box as (x1, y1, x2, y2, score); a score of -1 means
    nothing was detected. `pose_metas[i]["keypoints_body"]` are its body keypoints,
    normalised to the frame, with confidence. A frame with neither a box nor a confident
    keypoint gets an empty mask. With `temporal`, the previous frame's mask is carried into
    the decoder as a prior, which keeps an ambiguous boundary from flickering. Holes the
    mask encloses are filled and islands far smaller than the person are dropped."""
    N, H, W, _ = images.shape
    mm.load_model_gpu(model)
    device, dtype = mm.get_torch_device(), model.model.get_dtype()
    sam3 = model.model.diffusion_model
    sx, sy = SAM3_SIZE / W, SAM3_SIZE / H
    masks = torch.zeros(N, H, W)
    pbar = ProgressBar(N)
    prior, prior_box = None, None
    for i in range(N):
        box_inputs = point_inputs = None
        bbox = bboxes[i]
        detected = bbox is not None and bbox[-1] > 0
        if detected:
            box_inputs = torch.tensor([[[bbox[0] * sx, bbox[1] * sy], [bbox[2] * sx, bbox[3] * sy]]],
                                      device=device, dtype=dtype)
        kps = pose_metas[i]["keypoints_body"]
        points = [(kps[k][0], kps[k][1]) for k in PROMPT_KEYPOINTS if kps[k][2] > MIN_KEYPOINT_CONF]
        points += torso_points(kps, MIN_KEYPOINT_CONF)
        points = [(x * SAM3_SIZE, y * SAM3_SIZE) for x, y in points]
        if points:
            point_inputs = {"point_coords": torch.tensor([points], device=device, dtype=dtype),
                            "point_labels": torch.ones(1, len(points), dtype=torch.int32, device=device)}
        if box_inputs is None and point_inputs is None:
            prior, prior_box = None, None
            pbar.update(1)
            continue
        # the prior only carries over while the person stays where it was
        if prior is not None and not (detected and prior_box is not None and box_iou(bbox, prior_box) >= MIN_PRIOR_BOX_IOU):
            prior = None
        frame = common_upscale(images[i:i + 1, ..., :3].movedim(-1, 1), SAM3_SIZE, SAM3_SIZE, "bilinear", crop="disabled")
        frame = frame.to(device=device, dtype=dtype)
        with torch.inference_mode():
            logits = decode(sam3, frame, point_inputs, box_inputs, refine, mask_prior=prior)
            mask = F.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        masks[i] = torch.from_numpy(drop_islands(fill_holes((mask > 0).cpu().numpy().astype(np.uint8)))).float()
        prior, prior_box = (logits, bbox) if temporal and detected else (None, None)
        pbar.update(1)
    return masks
