# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import cv2
import numpy as np
import torch
from comfy import model_management as mm
from comfy.model_patcher import ModelPatcher

from ..pose_utils.pose2d_utils import box_convert_simple, keypoints_from_heatmaps, transform_preds
from .onnx_graph import GraphModule, OnnxGraph
from .vitpose import build_vitpose


def load_models(*models):
    """Bring the models to the compute device together, freeing VRAM held by other models
    if needed; ComfyUI moves them back out when another model needs the room."""
    mm.load_models_gpu([m.patcher for m in models], force_full_load=True)


class OnnxModel:
    """An ONNX model run with torch and managed by ComfyUI like any other model."""

    # tensors the graph returns: one heatmap / detection tensor, two for a SimCC pose head
    outputs = 1

    def __init__(self, checkpoint):
        graph = OnnxGraph(checkpoint)
        if len(graph.inputs) != 1 or len(graph.outputs) != self.outputs:
            raise ValueError(f"{checkpoint} has {len(graph.inputs)} inputs and {len(graph.outputs)} outputs; "
                             f"{type(self).__name__} takes one image and returns {self.outputs}")
        native = build_vitpose(graph)
        self.net = (native or GraphModule(graph)).eval()
        # [N, C, H, W] as the graph declares it; the batch dimension is usually a name
        self.input_shape = graph.input_shapes[graph.inputs[0]]
        # the graph executor runs the model's own Cast nodes, so it takes the declared input
        # type; the native ViTPose module skips them and takes its weights' type
        self.input_dtype = graph.input_dtypes[graph.inputs[0]] if native is None else next(self.net.parameters()).dtype
        if not self.input_dtype.is_floating_point:
            raise ValueError(f"{checkpoint} takes {self.input_dtype} input; the detection models are fed float images")
        self.patcher = ModelPatcher(self.net, load_device=mm.get_torch_device(), offload_device=mm.unet_offload_device())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def run(self, x):
        x = torch.from_numpy(np.ascontiguousarray(x)).to(self.patcher.load_device, self.input_dtype)
        with torch.inference_mode():
            out = self.net(x)
        if isinstance(out, tuple):
            return tuple(o.float().cpu().numpy() for o in out)
        return out.float().cpu().numpy()


class Yolo(OnnxModel):
    def __init__(self, checkpoint, threshold_conf=0.05, threshold_multi_persons=0.1, input_resolution=(640, 640), threshold_iou=0.5, threshold_bbox_shape_ratio=0.4, cat_id=[1], select_type='max', strict=True, sorted_func=None):
        super().__init__(checkpoint)

        self.input_width = 640
        self.input_height = 640

        self.threshold_multi_persons = threshold_multi_persons
        self.threshold_conf = threshold_conf
        self.threshold_iou = threshold_iou
        self.threshold_bbox_shape_ratio = threshold_bbox_shape_ratio
        self.input_resolution = input_resolution
        self.cat_id = cat_id
        self.select_type = select_type
        self.strict = strict
        self.sorted_func = sorted_func

    def postprocess(self, output, shape_raw, cat_id=[1]):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape

        outputs = np.squeeze(output)
        if len(outputs.shape) == 1:
            outputs = outputs[None]
        if output.shape[-1] != 6 and output.shape[1] == 84:
            outputs = np.transpose(outputs)

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = shape_raw[1] / self.input_width
        y_factor = shape_raw[0] / self.input_height

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        if outputs.shape[-1] == 6:
            max_scores = outputs[:, 4]
            classid = outputs[:, -1]

            threshold_conf_masks = max_scores >= self.threshold_conf
            classid_masks = classid[threshold_conf_masks] != 3.14159

            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[threshold_conf_masks][classid_masks]

            boxes = outputs[:, :4][threshold_conf_masks][classid_masks]
            boxes[:, [0, 2]] *= x_factor
            boxes[:, [1, 3]] *= y_factor
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            boxes = boxes.astype(np.int32)

        else:
            classes_scores = outputs[:, 4:]
            max_scores = np.amax(classes_scores, -1)
            threshold_conf_masks = max_scores >= self.threshold_conf

            classid = np.argmax(classes_scores[threshold_conf_masks], -1)

            classid_masks = classid!=3.14159

            classes_scores = classes_scores[threshold_conf_masks][classid_masks]
            max_scores = max_scores[threshold_conf_masks][classid_masks]
            classid = classid[classid_masks]

            xywh = outputs[:, :4][threshold_conf_masks][classid_masks]

            x = xywh[:, 0:1]
            y = xywh[:, 1:2]
            w = xywh[:, 2:3]
            h = xywh[:, 3:4]

            left = ((x - w / 2) * x_factor)
            top = ((y - h / 2) * y_factor)
            width = (w * x_factor)
            height = (h * y_factor)
            boxes = np.concatenate([left, top, width, height], axis=-1).astype(np.int32)

        boxes = boxes.tolist()
        scores = max_scores.tolist()
        class_ids = classid.tolist()

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.threshold_conf, self.threshold_iou)
        # Iterate over the selected indices after non-maximum suppression

        results = []
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = box_convert_simple(boxes[i], 'xywh2xyxy')
            score = scores[i]
            class_id = class_ids[i]
            results.append(box + [score] + [class_id])
            # # Draw the detection on the input image

        # Return the modified input image
        return np.array(results)


    def process_results(self, results, shape_raw, cat_id=[1], single_person=True):
        if isinstance(results, tuple):
            det_results = results[0]
        else:
            det_results = results

        person_results = []
        person_count = 0
        if len(results):
            max_idx = -1
            max_bbox_size = shape_raw[0] * shape_raw[1] * -10
            max_bbox_shape = -1

            bboxes = []
            idx_list = []
            for i in range(results.shape[0]):
                bbox = results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    idx_list.append(i)
                    bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                    if bbox_shape > max_bbox_shape:
                        max_bbox_shape = bbox_shape

            results = results[idx_list]

            for i in range(results.shape[0]):
                bbox = results[i]
                bboxes.append(bbox)
                if self.select_type == 'max':
                    bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1
                bbox_shape = max((bbox[2] - bbox[0]), ((bbox[3] - bbox[1])))
                if bbox_size > max_bbox_size:
                    if (self.strict or max_idx != -1) and bbox_shape < max_bbox_shape * self.threshold_bbox_shape_ratio:
                        continue
                    max_bbox_size = bbox_size
                    max_bbox_shape = bbox_shape
                    max_idx = i

            if self.sorted_func is not None and len(bboxes) > 0:
                max_idx = self.sorted_func(bboxes, shape_raw)
                bbox = bboxes[max_idx]
                if self.select_type == 'max':
                    max_bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                elif self.select_type == 'center':
                    max_bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1

            if max_idx != -1:
                person_count = 1

            if max_idx != -1:
                person = {}
                person['bbox'] = results[max_idx, :5]
                person['track_id'] = int(0)
                person_results.append(person)

            for i in range(results.shape[0]):
                bbox = results[i]
                if (bbox[-1] + 1 in cat_id) and (bbox[-2] > self.threshold_conf):
                    if self.select_type == 'max':
                        bbox_size = (bbox[2] - bbox[0]) * ((bbox[3] - bbox[1]))
                    elif self.select_type == 'center':
                        bbox_size = (abs((bbox[2] + bbox[0]) / 2 - shape_raw[1]/2)) * -1
                    if i != max_idx and bbox_size > max_bbox_size * self.threshold_multi_persons and bbox_size < max_bbox_size:
                        person_count += 1
                        if not single_person:
                            person = {}
                            person['bbox'] = results[i, :5]
                            person['track_id'] = int(person_count - 1)
                            person_results.append(person)
            # people the detector is at least fairly sure of; the 0.05 threshold above also
            # counts shadows and reflections
            strong = int(sum(1 for bbox in results if bbox[-2] >= 0.3))
            for person in person_results:
                person['person_count'] = strong
            return person_results
        else:
            return None


    def postprocess_threading(self, outputs, shape_raw, person_results, i, single_person=True, **kwargs):
        result = self.postprocess(outputs[i], shape_raw[i], cat_id=self.cat_id)
        result = self.process_results(result, shape_raw[i], cat_id=self.cat_id, single_person=single_person)
        if result is not None and len(result) != 0:
            person_results[i] = result


    def forward(self, img, shape_raw, **kwargs):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
            shape_raw = shape_raw.cpu().numpy()

        outputs = self.run(img)
        person_results = [[{'bbox': np.array([0., 0., 1.*shape_raw[i][1], 1.*shape_raw[i][0], -1]), 'track_id': -1}] for i in range(len(outputs))]

        for i in range(len(outputs)):
            self.postprocess_threading(outputs, shape_raw, person_results, i, **kwargs)
        return person_results


class ViTPose(OnnxModel):
    def forward(self, img, center, scale, **kwargs):
        heatmaps = self.run(img)
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                            center=center,
                                            scale=scale*200,
                                            unbiased=True,
                                            use_udp=False)
        return np.concatenate([points, prob], axis=2)


# RTMW's head is SimCC, not a heatmap: it classifies each keypoint's column and its row
# separately over the model input's pixels sampled SIMCC_SPLIT_RATIO times each, so the
# graph returns simcc_x [N, 133, 288 * 2] and simcc_y [N, 133, 384 * 2].
SIMCC_SPLIT_RATIO = 2.0
# What mmpose calls the SimCC score is min(max simcc_x, max simcc_y), and those are logits,
# not probabilities: measured over 25 frames of a dancer they run 1.8 to 8.0, so the raw
# number means nothing to the 0.3 / 0.5 thresholds the guard, the mask seeding and the
# drawing all apply to ViTPose's heatmap maxima. The rule for the divisor: a body keypoint
# the model has clearly found should read what the same keypoint reads on ViTPose. Those
# medians are 5.62 raw and 0.929, so the divisor is 6. It leaves the median body keypoint at
# 0.94, 99.8% of the body and 100% of the face above the 0.5 the drawing uses, and every
# body keypoint of a person who is in the crop above the guard's 0.3.
#
# What it cannot fix: SimCC is less decisive than a heatmap about a keypoint that is not
# there, because it still has to pick some column and some row inside the crop. On keypoints
# the crop cuts off, a third stay above 0.3 where only a sixth of ViTPose's do, so the guard
# sees a few more confident keypoints than it used to on frames that cut the person.
SIMCC_CONF_SCALE = 6.0


class RTMW(OnnxModel):
    """RTMW wholebody: the same 133 COCO-WholeBody keypoints as ViTPose, from a SimCC head,
    decoded the way mmpose's get_simcc_maximum / SimCCLabel.decode do."""

    outputs = 2

    def forward(self, img, center, scale, **kwargs):
        simcc_x, simcc_y = self.run(img)
        points = np.stack([simcc_x.argmax(axis=2), simcc_y.argmax(axis=2)], axis=-1)
        points = points.astype(np.float32) / SIMCC_SPLIT_RATIO
        vals = np.minimum(simcc_x.max(axis=2), simcc_y.max(axis=2))
        # the points are in the crop `crop` cut, so they go back to the frame through the
        # same transform the heatmap decode uses, over the input grid instead of a heatmap
        width, height = self.input_shape[3], self.input_shape[2]
        for i in range(len(points)):
            points[i] = transform_preds(points[i], center[i], scale[i] * 200, [width, height])
        prob = np.clip(vals / SIMCC_CONF_SCALE, 0.0, 1.0)
        return np.concatenate([points, prob[..., None]], axis=2)
