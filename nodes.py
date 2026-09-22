import os

import folder_paths

from . import log
from .guard import run_guard
from .models.download import download
from .models.onnx_models import RTMW, Yolo
from .models.sam3 import DEFAULT_SAM3
from .preprocess import detect, draw

# Models live in ComfyUI's own models/detection folder; the package only resolves full
# paths there (get_full_path does not filter by extension), so nothing else is needed.
_detection_path = os.path.join(folder_paths.models_dir, "detection")
folder_paths.add_model_folder_path("detection", _detection_path)

# The models the node runs, fetched on first use when they are not in models/detection.
# Each entry lists every file the model needs.
POSE = "rtmw_dw_x_l_wholebody_384x288.onnx"
YOLO = "yolov10x.onnx"
MODELS = {
    POSE: (
        ("rtmw_dw_x_l_wholebody_384x288.onnx", "https://huggingface.co/bukuroo/RTMW-ONNX/resolve/main/rtmw-l-384.onnx"),
    ),
    YOLO: (
        ("yolov10x.onnx", "https://huggingface.co/onnx-community/yolov10x/resolve/main/onnx/model.onnx"),
    ),
}


def detection_model_path(name):
    """Full path of a model in models/detection; when it (or one of its files) is missing it
    is downloaded first, next to the .onnx if that already exists."""
    found = folder_paths.get_full_path("detection", name)
    target_dir = os.path.dirname(found) if found else _detection_path
    os.makedirs(target_dir, exist_ok=True)
    for filename, url in MODELS[name]:
        if not os.path.isfile(os.path.join(target_dir, filename)):
            download(url, os.path.join(target_dir, filename))
    return folder_paths.get_full_path_or_raise("detection", name)


_detection_models = {"models": None}


def load_detection_models():
    """The RTMW and YOLO models, built once and kept."""
    if _detection_models["models"] is None:
        with log.step(f"building {POSE} and {YOLO}"):
            _detection_models["models"] = (RTMW(detection_model_path(POSE)), Yolo(detection_model_path(YOLO)))
    return _detection_models["models"]


class WanAnimatePreprocess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the body sticks in the pose images; 0 leaves the body out, -1 picks it from the frame size"}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the hand sticks in the pose images; 0 leaves the hands out, -1 picks it from the frame size"}),
                "draw_head": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw head keypoints"}),
                "face_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1, "tooltip": "When > 0, the detected face images are padded and resized to 512x512"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "POSEDATA")
    RETURN_NAMES = ("pose_images", "face_images", "mask", "pose_data")
    FUNCTION = "process"
    CATEGORY = "WanAnimate"
    DESCRIPTION = "The whole WanAnimate preprocess in one node: YOLOv10x finds the person, RTMW-l wholebody gives the keypoints, the face is cropped, SAM 3.1 segments the person from the box and keypoints, and the pose images are drawn at the frame size. The models are downloaded on first use. Feed it frames already at the generation size."

    def process(self, images, body_stick_width, hand_stick_width, draw_head, face_padding):
        pose_model, detector = load_detection_models()
        pose_data, face_images, mask = detect(detector, pose_model, images, face_padding=face_padding, sam3_model=DEFAULT_SAM3)
        pose_images = draw(pose_data, body_stick_width, hand_stick_width, draw_head)
        return (pose_images, face_images, mask, pose_data)


class WanAnimatePreprocessGuard:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "pose_data": ("POSEDATA",),
                "pose_guard": ("BOOLEAN", {"default": True, "tooltip": "Stop the workflow when a pose check fails (no detection, low confidence, torso jump, subject switch)"}),
                "mask_guard": ("BOOLEAN", {"default": True, "tooltip": "Stop the workflow when a mask check fails (empty, leaking, fragmented, keypoints outside, unstable)"}),
                "min_keypoint_conf": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Keypoints below this confidence are left out of the checks"}),
                "min_pose_conf": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "pose_low_confidence: mean body keypoint confidence below this"}),
                "max_torso_jump": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "pose_jump: torso keypoints moving more than this fraction of the box diagonal in one frame while the box stays"}),
                "min_mask_to_box": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "mask_empty: mask area below this fraction of the box area"}),
                "max_mask_outside_box": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "mask_leak: more than this fraction of the mask outside the box grown by 10%"}),
                "min_keypoint_recall": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "mask_missing_keypoints: fewer than this fraction of the confident keypoints inside the mask"}),
                "min_mask_iou": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "mask_unstable: mask IoU with the previous frame below this while the box IoU is above 0.7"}),
            },
        }

    RETURN_TYPES = ("MASK", "POSEDATA", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("mask", "pose_data", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = "WanAnimate"
    DESCRIPTION = "Beta. Checks the pose and the mask of WanAnimate Preprocess frame by frame (missing detections, pose glitches, empty / leaking / fragmented masks, keypoints outside the mask, unstable masks). Wire it between the preprocess and the sampler: a failed check of an enabled guard stops the workflow with the report; 'metrics' has every measurement per frame and 'timeline' plots them."

    def check(self, mask, pose_data, pose_guard, mask_guard, min_keypoint_conf, min_pose_conf, max_torso_jump,
              min_mask_to_box, max_mask_outside_box, min_keypoint_recall, min_mask_iou):
        thresholds = {
            "min_keypoint_conf": min_keypoint_conf, "min_pose_conf": min_pose_conf, "max_torso_jump": max_torso_jump,
            "min_mask_to_box": min_mask_to_box, "max_mask_outside_box": max_mask_outside_box,
            "min_keypoint_recall": min_keypoint_recall, "min_mask_iou": min_mask_iou,
        }
        frames = mask.shape[0] if mask.dim() == 3 else 1
        with log.step(f"checking {frames} frames (pose guard {'on' if pose_guard else 'off'}, mask guard {'on' if mask_guard else 'off'})"):
            report, passed, metrics, timeline = run_guard(mask, pose_data, thresholds, pose_guard, mask_guard)
        log.info(report.replace("\n", "\n    "))
        if not passed:
            raise RuntimeError(report)
        return (mask, pose_data, report, metrics, timeline)


NODE_CLASS_MAPPINGS = {
    "WanAnimatePreprocess": WanAnimatePreprocess,
    "WanAnimatePreprocessGuard": WanAnimatePreprocessGuard,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAnimatePreprocess": "WanAnimate Preprocess",
    "WanAnimatePreprocessGuard": "WanAnimate Preprocess Guard (beta)",
}
