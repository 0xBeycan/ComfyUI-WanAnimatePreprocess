"""Checks on the preprocess output: is the pose plausible and does the mask agree with it?

Every check is normalised by the person's size (the detector box) and, where possible,
crosses one signal with another that was produced independently: the SAM mask against the
ViTPose keypoints, the mask's motion against the box's motion. Pure geometry on the mask
cannot tell a stable wrong mask from a right one; the pose can.

Per frame, from `pose_data` (keypoints, detections) and the mask:

  no_detection        the detector found nobody, the whole frame was used as the box
  pose_low_confidence mean body-keypoint confidence below `min_pose_conf`
  pose_jump           torso keypoints moved more than `max_torso_jump` box diagonals in
                      one frame while the box hardly moved (pose glitch, not motion)
  subject_switch      box IoU with the previous frame below 0.3 (the detector picked
                      someone / something else)
  multi_person        the detector saw more than one person at 30% or more (warning only)
  mask_empty          mask area below `min_mask_to_box` of the box area
  mask_leak           more than `max_mask_outside_box` of the mask lies outside the box
                      grown by 10% (background or a neighbour pulled in)
  mask_fragmented     a second region at least 5% of the largest one (a ghost, a second
                      person, a split body); smaller detached pieces such as a shadow
                      blob are reported as mask_specks (warning only)
  mask_missing_keypoints
                      fewer than `min_keypoint_recall` of the confident keypoints fall
                      inside the mask (a missed limb, hand or foot); the report names them
  mask_unstable       mask IoU with the previous frame below `min_mask_iou` while the box
                      IoU is above 0.7 (the mask changed, the person did not)

The pose checks and the mask checks are switched on separately. Everything measured is
reported and plotted; a failed check of an enabled group stops the workflow, since
sampling on a wrong mask or pose is wasted. Thresholds are starting points: run with the
switches off on clips known to be good and bad and read `metrics` before trusting them.
"""
import json

import cv2
import numpy as np
import torch

BODY_NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
              "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear",
              "l_foot", "r_foot"]
TORSO = [0, 1, 2, 5, 8, 11]  # nose, neck, shoulders, hips: cannot jump a quarter of the body in one frame
WARNINGS = {"multi_person", "mask_specks"}
POSE_CHECKS = ("no_detection", "pose_low_confidence", "pose_jump", "subject_switch", "multi_person")
MASK_CHECKS = ("mask_empty", "mask_leak", "mask_fragmented", "mask_missing_keypoints", "mask_unstable", "mask_specks")
BOX_MARGIN = 0.10
SPECK_FRACTION = 0.01     # detached pieces above this fraction of the main region are reported
FRAGMENT_FRACTION = 0.05  # and above this one they count as a second object


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _ranges(frames):
    """[1, 2, 3, 7, 9, 10] -> '1-3, 7, 9-10'"""
    out, start, prev = [], None, None
    for f in frames:
        if start is None:
            start = prev = f
        elif f == prev + 1:
            prev = f
        else:
            out.append(f"{start}-{prev}" if prev > start else str(start))
            start = prev = f
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else str(start))
    return ", ".join(out)


def frame_metrics(masks, pose_metas, detections, min_keypoint_conf):
    """One dict of raw measurements per frame; thresholds are applied afterwards."""
    N, H, W = masks.shape
    rows = []
    prev = None
    for i in range(N):
        det = detections[i]
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        diag = float(np.hypot(bw, bh))
        kps = np.asarray(pose_metas[i]["keypoints_body"], dtype=np.float64).copy()
        kps[:, 0] *= W
        kps[:, 1] *= H
        confident = kps[:, 2] >= min_keypoint_conf
        mask = masks[i]
        area = int(mask.sum())

        m = {"frame": i, "detected": det["score"] > 0, "persons": det["persons"],
             "pose_conf": float(kps[:, 2].mean()), "confident_keypoints": int(confident.sum()),
             "mask_area": area / (H * W), "mask_to_box": area / (bw * bh)}

        # mask outside the grown box
        gx1, gy1 = int(max(0, x1 - BOX_MARGIN * bw)), int(max(0, y1 - BOX_MARGIN * bh))
        gx2, gy2 = int(min(W, x2 + BOX_MARGIN * bw)), int(min(H, y2 + BOX_MARGIN * bh))
        inside = int(mask[gy1:gy2, gx1:gx2].sum())
        m["mask_outside_box"] = (area - inside) / area if area else 0.0

        # detached regions, as fractions of the largest one
        if area:
            count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            areas = np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1]
            m["fragments"] = [round(float(a / areas[0]), 4) for a in areas[1:] if a >= areas[0] * SPECK_FRACTION]
        else:
            m["fragments"] = []

        # keypoints inside the (slightly grown) mask
        if area and confident.any():
            k = max(3, int(0.02 * diag) | 1)
            grown = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8))
            xs = np.clip(kps[:, 0].round().astype(int), 0, W - 1)
            ys = np.clip(kps[:, 1].round().astype(int), 0, H - 1)
            hit = grown[ys, xs].astype(bool) & confident
            m["keypoint_recall"] = float(hit.sum() / confident.sum())
            m["missed_keypoints"] = [BODY_NAMES[j] for j in np.flatnonzero(confident & ~hit)]
        else:
            m["keypoint_recall"] = 0.0 if confident.any() else 1.0
            m["missed_keypoints"] = [BODY_NAMES[j] for j in np.flatnonzero(confident)] if area == 0 else []

        # motion against the previous frame
        if prev is not None:
            m["box_iou_prev"] = _box_iou(det["bbox"], prev["bbox"]) if (m["detected"] and prev["detected"]) else None
            m["mask_iou_prev"] = _iou(mask, prev["mask"])
            both = confident & prev["confident"]
            torso = [j for j in TORSO if both[j]]
            m["torso_jump"] = float(np.linalg.norm(kps[torso, :2] - prev["kps"][torso, :2], axis=1).max() / diag) if torso else 0.0
        else:
            m["box_iou_prev"] = m["mask_iou_prev"] = None
            m["torso_jump"] = 0.0

        rows.append(m)
        prev = {"bbox": det["bbox"], "detected": m["detected"], "mask": mask, "kps": kps, "confident": confident}
    return rows


def apply_checks(rows, t):
    flags = {}

    def flag(name, i):
        flags.setdefault(name, []).append(i)

    for m in rows:
        i = m["frame"]
        if not m["detected"]:
            flag("no_detection", i)
        if m["pose_conf"] < t["min_pose_conf"]:
            flag("pose_low_confidence", i)
        if m["box_iou_prev"] is not None and m["box_iou_prev"] > 0.5 and m["torso_jump"] > t["max_torso_jump"]:
            flag("pose_jump", i)
        if m["box_iou_prev"] is not None and m["box_iou_prev"] < 0.3:
            flag("subject_switch", i)
        if m["persons"] > 1:
            flag("multi_person", i)
        if m["mask_to_box"] < t["min_mask_to_box"]:
            flag("mask_empty", i)
        elif m["mask_outside_box"] > t["max_mask_outside_box"]:
            flag("mask_leak", i)
        if any(f >= FRAGMENT_FRACTION for f in m["fragments"]):
            flag("mask_fragmented", i)
        elif m["fragments"]:
            flag("mask_specks", i)
        if m["keypoint_recall"] < t["min_keypoint_recall"] and m["mask_to_box"] >= t["min_mask_to_box"]:
            flag("mask_missing_keypoints", i)
        if (m["mask_iou_prev"] is not None and m["box_iou_prev"] is not None
                and m["box_iou_prev"] > 0.7 and m["mask_iou_prev"] < t["min_mask_iou"]):
            flag("mask_unstable", i)
    return flags


def write_report(rows, flags, enabled):
    """The report text and whether the enabled checks all passed."""
    failed = [name for name in flags if name in enabled and name not in WARNINGS]
    n = len(rows)
    lines = [f"Preprocess guard (beta): {'FAILED' if failed else 'passed'} - "
             f"{len(failed)} check(s) failed on {len({i for name in failed for i in flags[name]})}/{n} frames"]
    for name, frames in flags.items():
        kind = "warning" if name in WARNINGS else ("fail" if name in enabled else "off")
        line = f"- {name} ({kind}): {len(frames)} frame(s): {_ranges(frames)}"
        if name == "mask_missing_keypoints":
            missed = {}
            for i in frames:
                for k in rows[i]["missed_keypoints"]:
                    missed[k] = missed.get(k, 0) + 1
            line += " | missed: " + ", ".join(f"{k} x{c}" for k, c in sorted(missed.items(), key=lambda kv: -kv[1]))
        lines.append(line)
    return "\n".join(lines), not failed


def timeline_image(rows, flags):
    """Metrics over frames with the flagged frames marked, as an IMAGE tensor."""
    import matplotlib
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    n = len(rows)
    x = np.arange(n)
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, dpi=100)
    ax = axes[0]
    ax.plot(x, [m["mask_to_box"] for m in rows], label="mask / box area")
    ax.plot(x, [m["mask_outside_box"] for m in rows], label="mask outside box")
    ax.set_ylim(0, max(1.0, max(m["mask_to_box"] for m in rows) * 1.05))
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[1]
    ax.plot(x, [m["keypoint_recall"] for m in rows], label="keypoints inside mask")
    ax.plot(x, [m["pose_conf"] for m in rows], label="mean keypoint confidence")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[2]
    ax.plot(x, [m["mask_iou_prev"] if m["mask_iou_prev"] is not None else np.nan for m in rows], label="mask IoU vs previous")
    ax.plot(x, [m["box_iou_prev"] if m["box_iou_prev"] is not None else np.nan for m in rows], label="box IoU vs previous")
    ax.plot(x, [m["torso_jump"] for m in rows], label="torso jump / box diagonal")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[3]
    names = list(flags) or ["(no flags)"]
    for row, name in enumerate(names):
        frames = flags.get(name, [])
        ax.scatter(frames, [row] * len(frames), s=12, marker="s", color="tab:orange" if name in WARNINGS else "tab:red")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_ylim(-0.5, len(names) - 0.5)
    ax.set_xlabel("frame")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return torch.from_numpy(rgb).float().unsqueeze(0) / 255.0


def run_guard(mask, pose_data, thresholds, pose_guard, mask_guard):
    masks = (mask.cpu().numpy() > 0.5)
    pose_metas = pose_data["pose_metas_original"]
    detections = pose_data.get("detections")
    if detections is None:
        raise ValueError("pose_data has no per-frame detections; it must come from WanAnimate V1 Preprocess")
    if len(pose_metas) != masks.shape[0] or len(detections) != masks.shape[0]:
        raise ValueError(f"mask has {masks.shape[0]} frames, pose_data {len(pose_metas)}; connect the outputs of the same node")
    rows = frame_metrics(masks, pose_metas, detections, thresholds["min_keypoint_conf"])
    flags = apply_checks(rows, thresholds)
    enabled = set(POSE_CHECKS if pose_guard else ()) | set(MASK_CHECKS if mask_guard else ())
    report, passed = write_report(rows, flags, enabled)
    metrics = json.dumps({"thresholds": thresholds, "enabled": sorted(enabled), "flags": flags, "frames": rows})
    return report, passed, metrics, timeline_image(rows, flags)
