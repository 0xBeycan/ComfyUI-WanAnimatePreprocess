"""Person segmentation with SAM 3 / SAM 3.1, driven by the pose detector.

There is no text prompt. The person is described to SAM by what the pose pipeline already
knows about each frame: the detector's box and the body keypoints, as a box and positive
points on the joints, along the limbs and down the torso. That mask then propagates through
the tracker's memory, which is what carries the mask through motion blur, and the tracker
is re-seeded from a fresh prompt every `RESEED_INTERVAL` frames and whenever its mask stops
covering the keypoints - so the memory cannot drift away from the person the way a tracker
prompted only once does.

The model is ComfyUI's own SAM3 implementation loaded from `models/checkpoints`
(`sam3.1_multiplex_fp16.safetensors`, fetched on first use when missing), so it is loaded
and offloaded by ComfyUI like every other model.
"""
import contextlib
import io
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
R_SHOULDER, L_SHOULDER, R_HIP, L_HIP = 2, 5, 8, 11
# Keypoints sit on joints, so the body between them carries no positive evidence and the
# decoder drops parts of it: a close-up lost the lower half of a jacket, and a dancer's
# legs dropped out of the mask on the frames they moved fastest, while the knees the pose
# model was sure of sat outside it. These fractions along a limb or the torso put points
# on the body itself, halfway between the joints.
TORSO_FRACTIONS = (0.35, 0.65)
LIMB_FRACTIONS = (0.5,)
# Limbs to put a point on, as (from, to) keypoints: thighs, shins and upper arms.
LIMBS = ((R_HIP, 9), (L_HIP, 12), (9, 10), (12, 13), (R_SHOULDER, 3), (L_SHOULDER, 6))
MIN_KEYPOINT_CONF = 0.3
# How long the tracker may propagate before it is re-seeded from a fresh prompt. The
# tracker's own object score decays without reconditioning (the mask dies after a few
# hundred frames), so it is re-anchored well before that; the memory is what keeps the
# mask through frames the pose model loses to motion blur.
RESEED_INTERVAL = 24
# Re-seeding only helps from a frame the pose model is sure about: re-seeding on schedule
# landed on a motion-blurred frame, which replaced a good propagated mask with a bad
# prompt. A frame is an anchor when the detector found the person, enough keypoints are
# confident, they are confident on average, and the pose is no less complete than the one
# the current segment was seeded from - the blurred frames lose the knees while the head
# and the torso stay confident, so the count is what tells them apart.
MIN_ANCHOR_KEYPOINTS = 8
MIN_ANCHOR_CONF = 0.5
MIN_ANCHOR_COMPLETENESS = 0.9
# Propagating for longer than this without an anchor is not worth the drift risk: re-seed
# from whatever the frame offers.
MAX_PROPAGATE = 3 * RESEED_INTERVAL
# A propagated mask that stops covering this fraction of the frame's confident keypoints
# has come off the person: re-seed immediately.
MIN_TRACKED_RECALL = 0.9
# A binary mask handed to the tracker as logits: +/- this value.
MASK_LOGIT_SCALE = 10.0
# The decoder is told what the person is, never what it is not, and on a close-up in a
# cluttered room it annexes whatever is adjacent - a stuffed toy, a patch of wall. These
# many points of the previous frame's background, inside the box and this far (as a
# fraction of the box diagonal) from the mask it had, are given as negative prompts.
NEGATIVE_POINTS = 8
NEGATIVE_MARGIN = 0.04
# Where the mask logits are cut. Slightly below zero because the boundary is soft exactly
# where the thin parts are - loose hair, fingers, the edge of a foot - and they were left
# a few pixels outside the mask, which the block mask downstream then makes obvious.
MASK_THRESHOLD = -1.0
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


def body_points(kps, threshold, aspect=1.0):
    """Points on the body between the joints, in normalised coordinates: down the torso and
    along every limb whose two ends the pose model is sure of. `aspect` is W / H, needed
    where an x distance has to be stepped along y."""
    points = []
    if kps[R_SHOULDER][2] > threshold and kps[L_SHOULDER][2] > threshold:
        shoulder = ((kps[R_SHOULDER][0] + kps[L_SHOULDER][0]) / 2, (kps[R_SHOULDER][1] + kps[L_SHOULDER][1]) / 2)
        hips = [kps[i] for i in (R_HIP, L_HIP) if kps[i][2] > threshold]
        if hips:
            hip = (sum(h[0] for h in hips) / len(hips), sum(h[1] for h in hips) / len(hips))
            points += [(shoulder[0] + t * (hip[0] - shoulder[0]), shoulder[1] + t * (hip[1] - shoulder[1])) for t in TORSO_FRACTIONS]
        else:
            # the hips are out of frame: step down from the shoulders by their own width, in
            # frame units - the coordinates are normalised per axis, so the x span has to be
            # scaled by the aspect ratio before it can be added to y
            span = (abs(kps[R_SHOULDER][0] - kps[L_SHOULDER][0]) or 0.15) * aspect
            points += [(shoulder[0], min(shoulder[1] + t * span, 0.99)) for t in (1.0, 2.0)]
    for a, b in LIMBS:
        if kps[a][2] > threshold and kps[b][2] > threshold:
            points += [(kps[a][0] + t * (kps[b][0] - kps[a][0]), kps[a][1] + t * (kps[b][1] - kps[a][1])) for t in LIMB_FRACTIONS]
    return points


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


def decode(sam3, frame, point_inputs, box_inputs, refine):
    """Mask logits for one 1008x1008 frame from box and point prompts, with an optional
    refinement pass that feeds the first mask back to the decoder. This is
    SAM3Model.forward_segment with the image encoder run once: its features do not depend on
    the prompt, so the refinement pass only runs the SAM heads."""
    if not has_fast_path(sam3):
        logits = sam3.forward_segment(frame, point_inputs=point_inputs, box_inputs=box_inputs)
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
        backbone_features=backbone_feat, point_inputs=point_inputs, mask_inputs=None, box_inputs=box_inputs,
        high_res_features=high_res, multimask_output=(0 < num_pts <= 1))
    if refine:
        _, logits, _, _ = tracker._forward_sam_heads(
            backbone_features=backbone_feat, point_inputs=None, mask_inputs=logits, box_inputs=None,
            high_res_features=high_res, multimask_output=False)
    return logits


def background_points(previous_mask, bbox, W, H):
    """Points inside the box that the previous frame's mask says are background, kept clear
    of the mask by NEGATIVE_MARGIN of the box diagonal and spread over the box, in pixels."""
    if previous_mask is None or not previous_mask.any():
        return []
    x1, y1, x2, y2 = (int(max(0, min(v, limit))) for v, limit in zip(bbox[:4], (W, H, W, H)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return []
    inside = previous_mask[y1:y2, x1:x2]
    if not inside.any():
        return []
    margin = int(NEGATIVE_MARGIN * float(np.hypot(x2 - x1, y2 - y1)))
    free = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 3) > margin
    if not free.any():
        return []
    # spread the points: one per cell of a grid over the box, from the cells that have room
    rows = cols = int(np.ceil(np.sqrt(NEGATIVE_POINTS)))
    h, w = free.shape
    points = []
    for r in range(rows):
        for c in range(cols):
            cell = free[r * h // rows:(r + 1) * h // rows, c * w // cols:(c + 1) * w // cols]
            ys, xs = np.nonzero(cell)
            if len(ys):
                middle = len(ys) // 2
                points.append((x1 + c * w // cols + int(xs[middle]), y1 + r * h // rows + int(ys[middle])))
    return points[:NEGATIVE_POINTS]


def prompt_for(frame_index, bboxes, pose_metas, W, H, device, dtype, previous_mask=None):
    """(box_inputs, point_inputs) for the SAM decoder in its 1008x1008 space, or (None, None)
    when the frame has neither a detection nor a confident keypoint. The points are the
    keypoints and the body points as positives, plus background points from the previous
    frame's mask as negatives."""
    sx, sy = SAM3_SIZE / W, SAM3_SIZE / H
    bbox = bboxes[frame_index]
    box_inputs = None
    if bbox is not None and bbox[-1] > 0:
        box_inputs = torch.tensor([[[bbox[0] * sx, bbox[1] * sy], [bbox[2] * sx, bbox[3] * sy]]], device=device, dtype=dtype)
    kps = pose_metas[frame_index]["keypoints_body"]
    positive = [(kps[k][0] * W, kps[k][1] * H) for k in PROMPT_KEYPOINTS if kps[k][2] > MIN_KEYPOINT_CONF]
    positive += [(x * W, y * H) for x, y in body_points(kps, MIN_KEYPOINT_CONF, W / H)]
    negative = background_points(previous_mask, bbox, W, H) if bbox is not None and bbox[-1] > 0 else []
    point_inputs = None
    if positive:
        coords = [(x * sx, y * sy) for x, y in positive + negative]
        labels = [1] * len(positive) + [0] * len(negative)
        point_inputs = {"point_coords": torch.tensor([coords], device=device, dtype=dtype),
                        "point_labels": torch.tensor([labels], dtype=torch.int32, device=device)}
    return box_inputs, point_inputs


def keypoint_recall(mask, kps, W, H):
    """Fraction of the frame's confident body keypoints that fall inside `mask`, 1.0 when
    the pose model is sure of none of them."""
    confident = [(int(min(max(x * W, 0), W - 1)), int(min(max(y * H, 0), H - 1))) for x, y, c in kps if c > MIN_KEYPOINT_CONF]
    if not confident:
        return 1.0
    return sum(1 for x, y in confident if mask[y, x]) / len(confident)


def confident_count(pose_meta):
    return sum(1 for _, _, c in pose_meta["keypoints_body"] if c > MIN_KEYPOINT_CONF)


def is_anchor(frame_index, bboxes, pose_metas, reference_count=0):
    """Whether this frame's detection and pose are good enough to re-seed the tracker from.
    `reference_count` is how many keypoints the running segment was seeded with: a frame
    that sees fewer of them is a worse view of the person, not a better anchor."""
    bbox = bboxes[frame_index]
    if bbox is None or bbox[-1] <= 0:
        return False
    conf = [c for _, _, c in pose_metas[frame_index]["keypoints_body"] if c > MIN_KEYPOINT_CONF]
    return (len(conf) >= max(MIN_ANCHOR_KEYPOINTS, MIN_ANCHOR_COMPLETENESS * reference_count)
            and sum(conf) / len(conf) >= MIN_ANCHOR_CONF)


def clean(mask):
    """A person's mask: no enclosed holes, no islands far smaller than the body."""
    return drop_islands(fill_holes(mask.astype(np.uint8)))


def segment_frames(model, images, bboxes, pose_metas, refine=True, temporal=True, result=None):
    """[N, H, W] float masks of the detected person in `images` [N, H, W, 3].

    `bboxes[i]` is frame i's detector box as (x1, y1, x2, y2, score); a score of -1 means
    nothing was detected. `pose_metas[i]["keypoints_body"]` are its body keypoints,
    normalised to the frame, with confidence.

    A frame the detector and the pose model agree on is prompted with that box and those
    keypoints; the tracker's memory then propagates the mask, which is what carries it
    through frames the pose model loses to motion blur. The tracker is re-seeded from a
    fresh prompt every RESEED_INTERVAL frames - but only on a frame that is an anchor, and
    immediately when the propagated mask stops covering the keypoints. With `temporal` off,
    every frame is prompted on its own, without the tracker. `result`, if given, is filled
    with what happened for the log."""
    N, H, W, _ = images.shape
    mm.load_model_gpu(model)
    device, dtype = mm.get_torch_device(), model.model.get_dtype()
    sam3 = model.model.diffusion_model
    frames_chw = images[..., :3].movedim(-1, 1)
    masks = torch.zeros(N, H, W)
    pbar = ProgressBar(N)
    counts = {"prompted": 0, "propagated": 0, "re-seeded early": 0, "no prompt": 0, "empty prompt": 0}
    i = 0
    seed_frame, seed_count = -1, 0
    while i < N:
        previous = masks[i - 1].numpy() > 0.5 if i > 0 and masks[i - 1].any() else None
        box_inputs, point_inputs = prompt_for(i, bboxes, pose_metas, W, H, device, dtype, previous)
        if box_inputs is None or point_inputs is None:
            # nothing to describe the person with: carry the track on if there is one, else
            # leave this frame without a mask (the guard reports it as pose, not mask)
            counts["no prompt"] += 1
            if not (temporal and i > 0 and masks[i - 1].any()):
                pbar.update(1)
                i += 1
                continue
        else:
            frame = common_upscale(frames_chw[i:i + 1], SAM3_SIZE, SAM3_SIZE, "bilinear", crop="disabled").to(device, dtype)
            with torch.inference_mode():
                logits = decode(sam3, frame, point_inputs, box_inputs, refine)
                seed = (F.interpolate(logits.float(), size=(H, W), mode="bilinear",
                                      align_corners=False)[0, 0] > MASK_THRESHOLD).cpu().numpy()
            if seed.any():
                masks[i] = torch.from_numpy(clean(seed)).float()
                counts["prompted"] += 1
                seed_frame, seed_count = i, confident_count(pose_metas[i])
                pbar.update(1)
                i += 1
            else:
                # the decoder found nothing here; propagating from an empty mask is wasted work
                counts["empty prompt"] += 1
                if not (temporal and i > 0 and masks[i - 1].any()):
                    pbar.update(1)
                    i += 1
                    continue
        if not temporal or i >= N:
            continue
        # propagate from the last frame that has a mask; accept frames while the mask still
        # covers the keypoints, and stop at the first anchor once the interval is up
        while i < N:
            end = min(i + RESEED_INTERVAL, N)
            with torch.inference_mode():
                tracked = propagate(sam3, frames_chw[i - 1:end], masks[i - 1], device, dtype, H, W)
            if not tracked:
                break
            stop = False
            for k, mask in enumerate(tracked, start=i):
                if keypoint_recall(mask, pose_metas[k]["keypoints_body"], W, H) < MIN_TRACKED_RECALL:
                    counts["re-seeded early"] += 1
                    stop = True
                    break
                masks[k] = torch.from_numpy(clean(mask)).float()
                counts["propagated"] += 1
                pbar.update(1)
                i = k + 1
                since_seed = i - seed_frame if seed_frame >= 0 else RESEED_INTERVAL
                if since_seed >= RESEED_INTERVAL and i < N and is_anchor(i, bboxes, pose_metas, seed_count):
                    stop = True
                    break
                if since_seed >= MAX_PROPAGATE:
                    stop = True
                    break
            if stop:
                break
    if result is not None:
        result.update({k: v for k, v in counts.items() if v})
    log.info("segmented " + ", ".join(f"{v} frame(s) {k}" for k, v in counts.items() if v))
    return masks


def propagate(sam3, frames_chw, first_mask, device, dtype, H, W):
    """Masks for frames_chw[1:], propagated by the tracker's memory from `first_mask` on
    frames_chw[0]. Returns [] when this ComfyUI's tracker cannot be driven this way, which
    is decided on the first attempt and then remembered, so a version that does not support
    it costs one warning instead of one per frame."""
    if getattr(sam3, "_wanpre_no_propagation", False):
        return []
    initial = (first_mask[None, None].to(device, dtype) * 2 - 1) * MASK_LOGIT_SCALE
    try:
        with torch.inference_mode(), contextlib.redirect_stderr(io.StringIO()):
            # the core tracker prints a progress bar per call, which is one blank line per
            # segment in ComfyUI's log; our own step lines already report the progress
            result = sam3.forward_video(images=frames_chw, initial_masks=initial, text_prompts=None,
                                        max_objects=1, detect_interval=1, target_device=device, target_dtype=dtype)
            packed = result["packed_masks"]
            if packed is None:
                return []
            from comfy.ldm.sam3.tracker import unpack_masks
            tracked = unpack_masks(packed[:, 0]).float()[:, None]
            tracked = F.interpolate(tracked, size=(H, W), mode="bilinear", align_corners=False)[:, 0] > 0.5
            return [m.cpu().numpy() for m in tracked[1:]]
    except (AttributeError, ImportError, KeyError, TypeError, ValueError) as e:
        log.warning(f"this ComfyUI's SAM3 tracker cannot be propagated ({e}); prompting every frame instead")
        sam3._wanpre_no_propagation = True
        return []
