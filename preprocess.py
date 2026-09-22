"""The Wan Animate preprocess, frame by frame: person box (YOLO), body / hand / face
keypoints (ViTPose), face crops, the person mask (SAM3) and the drawn pose images."""
import cv2
import numpy as np
import torch
from comfy.utils import ProgressBar
from tqdm import tqdm

from . import log
from .models.onnx_models import load_models
from .models.sam3 import load_sam3, segment_frames
from .pose_utils.human_visualization import draw_aapose_by_meta_new
from .pose_utils.pose2d_utils import AAPoseMeta, bbox_from_detector, crop, load_pose_metas_from_kp2ds_seq
from .utils import get_face_bboxes

IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
POSE_INPUT_RESOLUTION = (256, 192)
POSE_CROP_RESCALE = 1.25
FACE_CROP_SCALE = 1.3
FACE_SIZE = 512
# A detector box that stops within this fraction of its own size from a frame edge belongs
# to a person the frame cuts off; it is extended to that edge before it prompts the mask,
# otherwise the decoder stops at the box and the clothing below it stays unmasked.
EDGE_SNAP = 0.15


def snap_to_frame(bbox, W, H):
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    bw, bh = x2 - x1, y2 - y1
    return np.array([0.0 if x1 < EDGE_SNAP * bw else x1, 0.0 if y1 < EDGE_SNAP * bh else y1,
                     float(W) if W - x2 < EDGE_SNAP * bw else x2, float(H) if H - y2 < EDGE_SNAP * bh else y2,
                     float(bbox[4])])


def detect(detector, pose_model, images, face_padding=0, sam3_model=None):
    """Runs the detector and the pose model on every frame of `images` [B, H, W, 3].

    Returns (pose_data, face_images, mask): pose_data carries the per-frame pose metas
    (`pose_metas` as AAPoseMeta for drawing, `pose_metas_original` as dicts with the
    normalised keypoints) and `detections` (the chosen person box as it prompts the mask,
    extended to frame edges it nearly touches, its score, -1 when nothing was detected and
    the whole frame was used, and the number of people the detector was fairly sure of);
    face_images are 512x512 crops around the face; mask is the SAM3 person mask from the
    checkpoint `sam3_model`, empty when None."""
    B, H, W, C = images.shape
    shape = np.array([H, W])[None]
    images_np = images.numpy()
    log.info(f"preprocessing {B} frames of {W}x{H}")
    with log.step("loading the detection models"):
        load_models(detector, pose_model)

    pbar = ProgressBar(B * 2)
    bboxes, person_counts = [], []
    result = {}
    with log.step(f"detecting the person on {B} frames", result):
        for i, img in enumerate(tqdm(images_np, desc="Detecting bboxes")):
            detection = detector(cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None], shape)[0][0]
            bbox, count = detection["bbox"], detection.get("person_count", 0)
            if bbox[-1] <= 0 or (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 10:
                # nothing usable detected: the pose, the mask prompt and the guard all see the
                # whole frame as the box, marked undetected
                bbox, count = np.array([0.0, 0.0, W, H, -1.0]), 0
            bboxes.append(bbox)
            person_counts.append(count)
            pbar.update_absolute(i + 1)
        result["frames without a person"] = sum(1 for b in bboxes if b[-1] <= 0)
        result["frames with several people"] = sum(1 for n in person_counts if n > 1)
    prompt_boxes = [snap_to_frame(b, W, H) for b in bboxes]

    kp2ds = []
    with log.step(f"extracting keypoints on {B} frames"):
        for i, (img, bbox) in enumerate(tqdm(zip(images_np, bboxes), total=B, desc="Extracting keypoints")):
            center, scale = bbox_from_detector(bbox, POSE_INPUT_RESOLUTION, rescale=POSE_CROP_RESCALE)
            img = crop(img, center, scale, POSE_INPUT_RESOLUTION)[0]
            img_norm = ((img - IMG_NORM_MEAN) / IMG_NORM_STD).transpose(2, 0, 1).astype(np.float32)
            kp2ds.append(pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None]))
            pbar.update_absolute(B + i + 1)
        pose_metas = load_pose_metas_from_kp2ds_seq(np.concatenate(kp2ds, 0), width=W, height=H)

    face_images = []
    result = {"fallback crops": 0}
    with log.step("cropping the faces", result):
        for i, meta in enumerate(pose_metas):
            x1, x2, y1, y2 = get_face_bboxes(meta["keypoints_face"][:, :2], scale=FACE_CROP_SCALE, image_shape=(H, W))
            if face_padding > 0:
                x1, y1 = max(0, x1 - face_padding), max(0, y1 - face_padding)
                x2, y2 = min(W, x2 + face_padding), min(H, y2 + face_padding)
            face = images_np[i][y1:y2, x1:x2]
            if face.size == 0:
                # no usable face box on this frame: fall back to the upper centre of the frame
                log.warning(f"empty face crop on frame {i}, using a centre crop instead")
                result["fallback crops"] += 1
                size = int(min(H, W) * 0.3)
                fx, fy = (W - size) // 2, int(H * 0.1)
                face = images_np[i][fy:fy + size, fx:fx + size]
                if face.size == 0:
                    face = np.zeros((size, size, C), dtype=images_np.dtype)
            face_images.append(cv2.resize(face, (FACE_SIZE, FACE_SIZE)))

    pose_data = {
        "pose_metas": [AAPoseMeta.from_humanapi_meta(meta) for meta in pose_metas],
        "pose_metas_original": pose_metas,
        "detections": [
            {"bbox": [float(v) for v in box[:4]], "score": float(box[4]), "persons": int(count)}
            for box, count in zip(prompt_boxes, person_counts)
        ],
    }
    if sam3_model is not None:
        result = {}
        with log.step(f"segmenting the person with {sam3_model} on {B} frames", result):
            mask = segment_frames(load_sam3(sam3_model), images, prompt_boxes, pose_metas)
            coverage = mask.mean(dim=(1, 2))
            result["empty masks"] = int((coverage == 0).sum())
            result["mask coverage"] = f"{coverage.min() * 100:.1f}-{coverage.max() * 100:.1f}%"
    else:
        log.info("no SAM3 checkpoint, the mask output is empty")
        mask = torch.zeros(B, H, W)
    return pose_data, torch.from_numpy(np.stack(face_images, 0)), mask


def draw(pose_data, body_stick_width=-1, hand_stick_width=-1, draw_head=True):
    """The pose images [B, H, W, 3] drawn from pose_data at the size of the frames the pose
    was found on, so they line up with the frames and the mask. A stick width of 0 leaves
    that part out."""
    pose_metas = pose_data["pose_metas"]
    pbar = ProgressBar(len(pose_metas))
    pose_images = []
    with log.step(f"drawing {len(pose_metas)} pose images"):
        for i, meta in enumerate(tqdm(pose_metas, desc="Drawing pose images")):
            canvas = np.zeros((meta.height, meta.width, 3), dtype=np.uint8)
            image = draw_aapose_by_meta_new(canvas, meta, draw_body=body_stick_width != 0, draw_hand=hand_stick_width != 0,
                                            draw_head=draw_head, body_stick_width=body_stick_width, hand_stick_width=hand_stick_width)
            pose_images.append(image)
            pbar.update_absolute(i + 1)
    return torch.from_numpy(np.stack(pose_images, 0)).float() / 255.0
