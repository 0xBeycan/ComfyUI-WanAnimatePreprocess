"""The guard on synthetic clips: a clean clip passes, each injected fault fires its own check
on the injected frames only. Runs without ComfyUI or any model:

    python -m pytest tests/test_guard.py
"""
import importlib.util
import json
import os

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("guard", os.path.join(ROOT, "guard.py"))
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

N, H, W = 40, 320, 240
THRESHOLDS = {"min_keypoint_conf": 0.3, "min_pose_conf": 0.35, "max_torso_jump": 0.25, "min_mask_to_box": 0.15,
              "max_mask_outside_box": 0.10, "min_keypoint_recall": 0.9, "min_mask_iou": 0.6}


def clip():
    """A person-sized rectangle drifting slowly across the frame, with keypoints inside it."""
    masks = torch.zeros(N, H, W)
    metas, detections = [], []
    for i in range(N):
        x1, y1 = 60 + i, 40
        x2, y2 = x1 + 100, y1 + 240
        masks[i, y1:y2, x1:x2] = 1.0
        # 20 body keypoints spread inside the rectangle, all confident
        xs = np.linspace(x1 + 10, x2 - 10, 5)
        ys = np.linspace(y1 + 10, y2 - 10, 4)
        pts = np.array([(x, y, 0.9) for y in ys for x in xs])
        pts[:, 0] /= W
        pts[:, 1] /= H
        metas.append({"width": W, "height": H, "keypoints_body": pts})
        detections.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "score": 0.95, "persons": 1})
    return masks, {"pose_metas_original": metas, "detections": detections}


def run(masks, pose_data, pose_guard=True, mask_guard=True):
    report, passed, metrics, timeline = guard.run_guard(masks, pose_data, THRESHOLDS, pose_guard, mask_guard)
    flags = json.loads(metrics)["flags"]
    assert timeline.shape[0] == 1 and timeline.shape[-1] == 3
    return passed, flags, report


def fails_only(flags, name, frames):
    failing = {k: v for k, v in flags.items() if k not in guard.WARNINGS}
    assert list(failing) == [name] or set(failing) - {name} <= {"mask_unstable"}, failing
    assert failing[name] == frames


def test_clean_clip_passes():
    masks, pose_data = clip()
    passed, flags, report = run(masks, pose_data)
    assert passed and not {k for k in flags if k not in guard.WARNINGS}, report


def test_empty_start_is_mask_empty():
    masks, pose_data = clip()
    masks[:10] = 0
    passed, flags, _ = run(masks, pose_data)
    assert not passed
    fails_only(flags, "mask_empty", list(range(10)))


def test_mask_dying_mid_clip():
    masks, pose_data = clip()
    masks[25:] = 0
    passed, flags, _ = run(masks, pose_data)
    assert not passed and flags["mask_empty"] == list(range(25, N))


def test_cut_limb_names_the_keypoints():
    masks, pose_data = clip()
    masks[20:25, 200:, :] = 0  # the lowest keypoint row sits at y ~ 270
    passed, flags, report = run(masks, pose_data)
    assert not passed
    assert flags["mask_missing_keypoints"] == list(range(20, 25))
    assert "missed:" in report


def test_leak_outside_box():
    masks, pose_data = clip()
    masks[30, :, 200:] = 1.0  # a stripe far from the person
    passed, flags, _ = run(masks, pose_data)
    assert not passed and 30 in flags["mask_leak"] and 30 in flags["mask_fragmented"]


def test_missing_detection_is_pose_check():
    masks, pose_data = clip()
    pose_data["detections"][5] = {"bbox": [0.0, 0.0, float(W), float(H)], "score": -1.0, "persons": 0}
    passed, flags, _ = run(masks, pose_data)
    assert not passed and flags["no_detection"] == [5]
    # with the pose guard off the same clip passes (the mask checks are still fine)
    passed, _, _ = run(masks, pose_data, pose_guard=False)
    assert passed


def test_subject_switch():
    masks, pose_data = clip()
    pose_data["detections"][15]["bbox"] = [0.0, 0.0, 40.0, 40.0]
    passed, flags, _ = run(masks, pose_data)
    assert not passed and 15 in flags["subject_switch"]


def test_torso_jump():
    masks, pose_data = clip()
    pts = pose_data["pose_metas_original"][12]["keypoints_body"].copy()
    pts[guard.TORSO, 0] += 0.5
    pose_data["pose_metas_original"][12]["keypoints_body"] = pts
    passed, flags, _ = run(masks, pose_data)
    assert not passed and 12 in flags["pose_jump"]


def test_low_confidence():
    masks, pose_data = clip()
    pts = pose_data["pose_metas_original"][8]["keypoints_body"].copy()
    pts[:, 2] = 0.1
    pose_data["pose_metas_original"][8]["keypoints_body"] = pts
    passed, flags, _ = run(masks, pose_data)
    assert not passed and flags["pose_low_confidence"] == [8]


def test_guards_off_never_raise_but_still_report():
    masks, pose_data = clip()
    masks[:10] = 0
    passed, flags, report = run(masks, pose_data, pose_guard=False, mask_guard=False)
    assert passed and flags["mask_empty"] == list(range(10)) and "(off)" in report


def test_frame_count_mismatch_is_an_error():
    masks, pose_data = clip()
    with pytest.raises(ValueError):
        guard.run_guard(masks[:5], pose_data, THRESHOLDS, True, True)


def test_foreign_pose_data_is_an_error():
    masks, _ = clip()
    with pytest.raises(ValueError, match="WanAnimate Preprocess"):
        guard.run_guard(masks, {"something": 1}, THRESHOLDS, True, True)


def test_resized_mask_is_an_error():
    masks, pose_data = clip()
    small = torch.nn.functional.interpolate(masks[None], size=(H // 2, W // 2))[0]
    with pytest.raises(ValueError, match="before any resize"):
        guard.run_guard(small, pose_data, THRESHOLDS, True, True)


def test_single_frame_2d_mask_is_accepted():
    masks, pose_data = clip()
    pose_data = {"pose_metas_original": pose_data["pose_metas_original"][:1], "detections": pose_data["detections"][:1]}
    passed, flags, _ = run(masks[0], pose_data)
    assert passed
