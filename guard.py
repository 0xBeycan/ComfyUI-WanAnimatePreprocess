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


PANEL_W, PANEL_H, MARGIN_L, MARGIN_R, MARGIN_T, GAP = 1200, 190, 210, 20, 34, 34
COLORS = {"blue": (31, 119, 180), "orange": (255, 127, 14), "green": (44, 160, 44), "red": (214, 39, 40), "grey": (150, 150, 150)}


def _polyline(img, values, x0, y0, w, h, color):
    pts = [(int(x0 + i / max(len(values) - 1, 1) * w), int(y0 + h - min(max(v, 0.0), 1.0) * h))
           for i, v in enumerate(values) if v is not None]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, color, 1, cv2.LINE_AA)


def _panel(img, y0, title, series):
    """One panel: a 0..1 axis with gridlines and the named series drawn over it."""
    x0, w, h = MARGIN_L, PANEL_W - MARGIN_L - MARGIN_R, PANEL_H - GAP
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), COLORS["grey"], 1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(y0 + h - frac * h)
        if 0 < frac < 1:
            cv2.line(img, (x0, y), (x0 + w, y), (225, 225, 225), 1)
        cv2.putText(img, f"{frac:.2f}", (x0 - 38, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["grey"], 1, cv2.LINE_AA)
    # title and legend on the line above the panel
    cv2.putText(img, title, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    legend_x = x0 + 90
    for name, values, color in series:
        _polyline(img, values, x0, y0, w, h, color)
        cv2.line(img, (legend_x, y0 - 14), (legend_x + 18, y0 - 14), color, 2)
        cv2.putText(img, name, (legend_x + 24, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
        legend_x += 8 * len(name) + 56


def timeline_image(rows, flags):
    """Metrics over frames with the flagged frames marked, as an IMAGE tensor (no plotting
    library needed, drawn with OpenCV)."""
    n = len(rows)
    names = list(flags) or ["(no flags)"]
    flag_h = 24 * len(names) + 40
    height = MARGIN_T + 3 * PANEL_H + flag_h + 30
    img = np.full((height, PANEL_W, 3), 255, np.uint8)
    y = MARGIN_T
    _panel(img, y, "mask", [("mask / box area", [m["mask_to_box"] for m in rows], COLORS["blue"]),
                            ("mask outside box", [m["mask_outside_box"] for m in rows], COLORS["orange"])])
    y += PANEL_H
    _panel(img, y, "pose", [("keypoints inside mask", [m["keypoint_recall"] for m in rows], COLORS["blue"]),
                            ("mean keypoint confidence", [m["pose_conf"] for m in rows], COLORS["orange"])])
    y += PANEL_H
    _panel(img, y, "motion", [("mask IoU vs previous", [m["mask_iou_prev"] for m in rows], COLORS["blue"]),
                              ("box IoU vs previous", [m["box_iou_prev"] for m in rows], COLORS["orange"]),
                              ("torso jump / box diagonal", [m["torso_jump"] for m in rows], COLORS["green"])])
    y += PANEL_H
    x0, w = MARGIN_L, PANEL_W - MARGIN_L - MARGIN_R
    cv2.putText(img, "flags", (x0, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y), (x0 + w, y + flag_h - 30), COLORS["grey"], 1)
    for row, name in enumerate(names):
        cy = y + 16 + 24 * row
        cv2.putText(img, name, (12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
        color = COLORS["orange"] if name in WARNINGS else COLORS["red"]
        for f in flags.get(name, []):
            cx = int(x0 + f / max(n - 1, 1) * w)
            cv2.rectangle(img, (cx - 2, cy - 4), (cx + 2, cy + 4), color, -1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        cx = int(x0 + frac * w)
        cv2.putText(img, str(int(frac * (n - 1))), (cx - 8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["grey"], 1, cv2.LINE_AA)
    cv2.putText(img, "frame", (PANEL_W // 2 - 20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1, cv2.LINE_AA)
    return torch.from_numpy(img).float().unsqueeze(0) / 255.0


def run_guard(mask, pose_data, thresholds, pose_guard, mask_guard):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"mask must be [frames, height, width], got a tensor of shape {tuple(mask.shape)}")
    masks = (mask.cpu().numpy() > 0.5)
    pose_metas = pose_data.get("pose_metas_original") if isinstance(pose_data, dict) else None
    detections = pose_data.get("detections") if isinstance(pose_data, dict) else None
    if pose_metas is None or detections is None:
        raise ValueError("pose_data has no per-frame keypoints and detections; it must come from WanAnimate Preprocess")
    N, H, W = masks.shape
    if len(pose_metas) != N or len(detections) != N:
        raise ValueError(f"mask has {N} frames, pose_data {len(pose_metas)}; connect the outputs of the same node")
    if pose_metas and (pose_metas[0]["height"], pose_metas[0]["width"]) != (H, W):
        raise ValueError(f"mask is {W}x{H} but the pose was found on {pose_metas[0]['width']}x{pose_metas[0]['height']} "
                         "frames; connect the mask straight from WanAnimate Preprocess, before any resize")
    rows = frame_metrics(masks, pose_metas, detections, thresholds["min_keypoint_conf"])
    flags = apply_checks(rows, thresholds)
    enabled = set(POSE_CHECKS if pose_guard else ()) | set(MASK_CHECKS if mask_guard else ())
    report, passed = write_report(rows, flags, enabled)
    metrics = json.dumps({"thresholds": thresholds, "enabled": sorted(enabled), "flags": flags, "frames": rows})
    return report, passed, metrics, timeline_image(rows, flags)
