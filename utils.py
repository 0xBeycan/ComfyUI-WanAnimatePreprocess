# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import numpy as np


def get_face_bboxes(kp2ds, scale, image_shape):
    h, w = image_shape
    kp2ds_face = kp2ds.copy()[1:] * (w, h)

    # Drop NaN/inf keypoints (undetected face); fall back to image center if none remain.
    kp2ds_face = kp2ds_face[np.isfinite(kp2ds_face).all(axis=1)]
    if len(kp2ds_face) == 0:
        kp2ds_face = np.array([[w / 2.0, h / 2.0]])

    min_x, min_y = np.min(kp2ds_face, axis=0)
    max_x, max_y = np.max(kp2ds_face, axis=0)

    initial_width = max_x - min_x
    initial_height = max_y - min_y

    # Degenerate (collapsed) axis gives 0/0 -> NaN below; clamp to a minimum box around its center.
    min_side = max(8.0, 0.05 * min(h, w))
    if not np.isfinite(initial_width) or initial_width < min_side:
        cx = (min_x + max_x) / 2
        min_x, max_x = cx - min_side / 2, cx + min_side / 2
        initial_width = max_x - min_x
    if not np.isfinite(initial_height) or initial_height < min_side:
        cy = (min_y + max_y) / 2
        min_y, max_y = cy - min_side / 2, cy + min_side / 2
        initial_height = max_y - min_y

    initial_area = initial_width * initial_height

    expanded_area = initial_area * scale

    new_width = np.sqrt(expanded_area * (initial_width / initial_height))
    new_height = np.sqrt(expanded_area * (initial_height / initial_width))

    delta_width = (new_width - initial_width) / 2
    delta_height = (new_height - initial_height) / 4

    expanded_min_x = max(min_x - delta_width, 0)
    expanded_max_x = min(max_x + delta_width, w)
    expanded_min_y = max(min_y - 3 * delta_height, 0)
    expanded_max_y = min(max_y + delta_height, h)

    return [int(expanded_min_x), int(expanded_max_x), int(expanded_min_y), int(expanded_max_y)]
