# ComfyUI-WanAnimatePreprocess

The preprocess for [Wan 2.2 Animate](https://github.com/Wan-Video/Wan2.2/tree/main/wan/modules/animate/preprocess)
in one node: the person's pose, face crops, mask and the drawn pose images, frame by frame,
plus a guard that stops the workflow when the pose or the mask looks wrong.

Everything runs with torch on whatever device ComfyUI uses (CUDA, ROCm, MPS or CPU). There
is no onnxruntime, so nothing has to match the CUDA version torch was built with, and the
models take part in ComfyUI's memory management like any other model.

## Nodes

### WanAnimate Preprocess

Inputs: the frames (`images`), the model choices (`vitpose_model`, `yolo_model`,
`sam3_model`), the generation size (`width`, `height`) and the drawing options
(`body_stick_width`, `hand_stick_width`, `draw_head`, `face_padding`).

Outputs: `pose_images` (the drawn pose at the generation size), `face_images` (512x512
crops around the face), `mask` (the person, every frame) and `pose_data` (for the guard).

Per frame: YOLO finds the person, ViTPose gives body / hand / face keypoints from that box,
the face is cropped around the face keypoints, and SAM3 segments the person from the box
and the body keypoints given to its decoder as a box and positive points. There is no text
prompt and no tracker: frames are independent, so nothing drifts or ghosts over long clips,
and each mask is as good as that frame's detection. Islands far smaller than the main region
are dropped. With `sam3_model` set to `none` the mask output is empty.

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

The model widgets list the defaults before they exist and download them on the first run
that selects them:

- `yolov10x.onnx` and `vitpose_h_wholebody_model.onnx` (+ `vitpose_h_wholebody_data.bin`)
  into `ComfyUI/models/detection`, from
  [onnx-community/yolov10x](https://huggingface.co/onnx-community/yolov10x/blob/main/onnx/model.onnx)
  (saved under the name above) and
  [Kijai/vitpose_comfy](https://huggingface.co/Kijai/vitpose_comfy/tree/main/onnx), about 2.7 GB
- `sam3.1_multiplex_fp16.safetensors` into `ComfyUI/models/checkpoints`, from
  [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1), 1.7 GB

Other files in those folders are listed too. Any fp32 or fp16 ONNX export of these model
families loads: YOLOv10 n / s / m / x (also exports with the raw `[1, 84, N]` output),
ViTPose wholebody s / b / l / h. ViTPose graphs become a native torch module (fused
LayerNorm, GELU and scaled-dot-product attention); everything else runs through a generic
ONNX-op executor (`models/onnx_graph.py`) covering the standard op set of CNN and ViT
exports. Quantized exports are not supported; an unsupported op is reported when the model
is loaded, naming the op. SAM3 is ComfyUI's own implementation, so any SAM 3 / 3.1
checkpoint the core loader accepts works.

## Tests

`python -m pytest tests` runs the guard on synthetic clips (no models needed).
`tests/compare_onnxruntime.py` checks the torch executor against onnxruntime on a model
file (install onnxruntime for it).
