# ComfyUI-WanAnimatePreprocess

The preprocess for [Wan 2.2 Animate](https://github.com/Wan-Video/Wan2.2/tree/main/wan/modules/animate/preprocess)
in one node: the person's pose, face crops, mask and the drawn pose images, frame by frame,
plus a guard that stops the workflow when the pose or the mask looks wrong.

Everything runs with torch on whatever device ComfyUI uses (CUDA, ROCm, MPS or CPU). There
is no onnxruntime, so nothing has to match the CUDA version torch was built with, and the
models take part in ComfyUI's memory management like any other model.

## Nodes

### WanAnimate Preprocess

Inputs: the frames (`images`) and the drawing options (`body_stick_width`,
`hand_stick_width`, `draw_head`, `face_padding`). The models are fixed (YOLOv10x,
SDPose-Wholebody, SAM 3.1 multiplex) and downloaded on first use.

Outputs: `pose_images` (the drawn pose, at the frame size so it lines up with the frames
and the mask - feed the node frames already at the generation size), `face_images`
(512x512 crops around the face), `mask` (the person, every frame) and `pose_data` (for
the guard).

Per frame: YOLO finds the person, SDPose gives body / hand / face keypoints from that box,
the face is cropped around the face keypoints, and SAM3 segments the person from the box
and the body keypoints given to its decoder as a box and positive points. A box that stops
just short of a frame edge is extended to it, so a person the frame cuts off is masked down
to the edge. There is no text prompt and no tracker: frames are independent, so nothing
drifts or ghosts over long clips, and each mask is as good as that frame's detection.
Islands far smaller than the main region are dropped.

The console shows every step with its duration and what it produced (frames without a
person, fallback face crops, empty masks, mask coverage).

### WanAnimate Preprocess Guard (beta)

Sits between the preprocess and the sampler: takes `mask` and `pose_data`, passes both
through, and stops the workflow with a report when a check of an enabled guard fails,
because sampling on a bad mask or pose is wasted. Every check is normalised by the detected
person's box and crosses the mask with the independently produced pose:

- `pose_guard`: `no_detection`, `pose_low_confidence`, `pose_jump` (torso moving in one
  frame while the box stays), `subject_switch` (the detector picked something else);
  `multi_person` is a warning
- `mask_guard`: `mask_empty`, `mask_leak` (outside the grown box), `mask_fragmented` (a
  second region at least 5% of the main one), `mask_missing_keypoints` (confident keypoints
  outside the mask, named in the report), `mask_unstable` (the mask changed, the person did
  not); `mask_specks` is a warning

`report` is the text that also goes to the console, `metrics` is JSON with every
measurement per frame and `timeline` plots them. Switch both guards off to calibrate the
thresholds on clips you know: nothing stops then and the report, metrics and timeline are
still produced.

## Models

Downloaded on the first run that needs them:

- `yolov10x.onnx` into `ComfyUI/models/detection`, from
  [onnx-community/yolov10x](https://huggingface.co/onnx-community/yolov10x/blob/main/onnx/model.onnx)
  (saved under the name above), 113 MB
- `sdpose_wholebody_fp16.safetensors` into `ComfyUI/models/checkpoints`, from
  [Comfy-Org/SDPose](https://huggingface.co/Comfy-Org/SDPose), 1.9 GB
- `sam3.1_multiplex_fp16.safetensors` into `ComfyUI/models/checkpoints`, from
  [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1), 1.7 GB

YOLO runs through this package's own generic ONNX-op executor (`models/onnx_graph.py`).
SDPose and SAM3 are ComfyUI's own implementations, loaded as ComfyUI checkpoints, so they
take part in its memory management; SDPose needs ComfyUI 0.36 or newer.

## Tests

`python -m pytest tests` runs the guard on synthetic clips (no models needed).
`tests/compare_onnxruntime.py` checks the torch executor against onnxruntime on a model
file (install onnxruntime for it).
