import os

import folder_paths

from . import log
from .guard import run_guard
from .models.download import download
from .models.onnx_models import ViTPose, Yolo
from .models.sam3 import DEFAULT_SAM3, sam3_choices
from .preprocess import detect, draw

_detection_path = os.path.join(folder_paths.models_dir, "detection")
folder_paths.add_model_folder_path("detection", _detection_path)
# Newer ComfyUI registers "detection" itself with supported_pt_extensions, which has no
# ".onnx", so add_model_folder_path only appends the path and every .onnx gets filtered out.
# An empty set means no filter is applied at all, leave that case alone.
_detection_exts = folder_paths.folder_names_and_paths["detection"][1]
if _detection_exts and ".onnx" not in _detection_exts:
    _detection_exts.add(".onnx")
    # Widening the set does not invalidate anything that already cached the filtered
    # list, and INPUT_TYPES would keep serving the stale .bin-only result. Both cache
    # layers are internals, so tolerate them being renamed or dropped upstream.
    _list_cache = getattr(folder_paths, "filename_list_cache", None)
    if isinstance(_list_cache, dict):
        _list_cache.pop("detection", None)
    _cache_helper = getattr(folder_paths, "cache_helper", None)
    if hasattr(_cache_helper, "clear"):
        _cache_helper.clear()

# The default models, fetched on first use when they are not in models/detection. Each
# entry lists every file the model needs.
DEFAULT_VITPOSE = "vitpose_h_wholebody_model.onnx"
DEFAULT_YOLO = "yolov10x.onnx"
DEFAULT_MODELS = {
    DEFAULT_VITPOSE: (
        ("vitpose_h_wholebody_model.onnx", "https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_model.onnx"),
        ("vitpose_h_wholebody_data.bin", "https://huggingface.co/Kijai/vitpose_comfy/resolve/main/onnx/vitpose_h_wholebody_data.bin"),
    ),
    DEFAULT_YOLO: (
        ("yolov10x.onnx", "https://huggingface.co/onnx-community/yolov10x/resolve/main/onnx/model.onnx"),
    ),
}


def detection_model_choices():
    """Files in models/detection plus the defaults, which are listed before they exist so a
    workflow can select them and have them downloaded on its first run."""
    return sorted(set(folder_paths.get_filename_list("detection")) | set(DEFAULT_MODELS))


def detection_model_path(name):
    """Full path of a model in models/detection; a default model that is missing (or missing
    one of its files) is downloaded first, next to the .onnx if that already exists."""
    files = DEFAULT_MODELS.get(name, ())
    if files:
        found = folder_paths.get_full_path("detection", name)
        target_dir = os.path.dirname(found) if found else _detection_path
        os.makedirs(target_dir, exist_ok=True)
        for filename, url in files:
            if not os.path.isfile(os.path.join(target_dir, filename)):
                download(url, os.path.join(target_dir, filename))
    return folder_paths.get_full_path_or_raise("detection", name)


_detection_models = {"names": None, "models": None}


def load_detection_models(vitpose_model, yolo_model):
    """The ViTPose and YOLO models, built once and kept for the same selection."""
    if _detection_models["names"] == (vitpose_model, yolo_model):
        return _detection_models["models"]
    _detection_models["names"], _detection_models["models"] = None, None
    with log.step(f"building {vitpose_model} and {yolo_model}"):
        models = (ViTPose(detection_model_path(vitpose_model)), Yolo(detection_model_path(yolo_model)))
    _detection_models["names"], _detection_models["models"] = (vitpose_model, yolo_model), models
    return models


class WanAnimatePreprocess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "vitpose_model": (detection_model_choices(), {"default": DEFAULT_VITPOSE, "tooltip": f"Loaded from the 'ComfyUI/models/detection' folder; {DEFAULT_VITPOSE} is downloaded on first use when missing"}),
                "yolo_model": (detection_model_choices(), {"default": DEFAULT_YOLO, "tooltip": f"Loaded from the 'ComfyUI/models/detection' folder; {DEFAULT_YOLO} is downloaded on first use when missing"}),
                "sam3_model": (sam3_choices(), {"default": DEFAULT_SAM3, "tooltip": "SAM 3 / 3.1 checkpoint from 'ComfyUI/models/checkpoints' (sam3.1_multiplex_fp16.safetensors is downloaded on first use when missing). Every frame's person is segmented from its detected bbox and body keypoints, no text prompt and no tracking; 'none' leaves the mask output empty"}),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 1, "tooltip": "Width of the generation"}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 1, "tooltip": "Height of the generation"}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the body sticks. Set to 0 to disable body drawing, -1 for auto"}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the hand sticks. Set to 0 to disable hand drawing, -1 for auto"}),
                "draw_head": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw head keypoints"}),
                "face_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1, "tooltip": "When > 0, the detected face images are padded and resized to 512x512"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "POSEDATA")
    RETURN_NAMES = ("pose_images", "face_images", "mask", "pose_data")
    FUNCTION = "process"
    CATEGORY = "WanAnimate"
    DESCRIPTION = "The whole WanAnimate preprocess in one node: loads the selected ViTPose, YOLO and SAM3 models (downloading the defaults when missing), detects the person's pose and face on every frame, segments the person from its bbox and keypoints, and draws the pose images."

    def process(self, images, vitpose_model, yolo_model, sam3_model, width, height, body_stick_width, hand_stick_width,
                draw_head, face_padding):
        pose_model, detector = load_detection_models(vitpose_model, yolo_model)
        pose_data, face_images, mask = detect(detector, pose_model, images, face_padding=face_padding, sam3_model=sam3_model)
        pose_images = draw(pose_data, width, height, body_stick_width, hand_stick_width, draw_head)
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
        with log.step(f"checking {mask.shape[0]} frames (pose guard {'on' if pose_guard else 'off'}, mask guard {'on' if mask_guard else 'off'})"):
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
