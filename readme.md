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
RTMW-l wholebody at 384x288, SAM 3.1 multiplex) and downloaded on first use.

Outputs: `pose_images` (the drawn pose, at the frame size so it lines up with the frames
and the mask - feed the node frames already at the generation size), `face_images`
(512x512 crops around the face), `mask` (the person, every frame) and `pose_data` (for
the guard).

Per frame: YOLO finds the person, RTMW gives body / hand / face keypoints from the crop
of that box, and the face is cropped around the face keypoints. The detector's box is
widened to the boxes of the frames around it and extended to a frame edge it nearly touches,
because it shrinks to the upper body on a close-up and around the blur on a fast frame,
which is what used to cut the legs and the feet out of both the pose crop and the mask.

SAM 3.1 then segments the person: a frame the detector and the pose model agree on is
prompted with that box and with points on the joints, along every limb and down the torso
(the joints alone leave dark clothing without positive evidence), and the tracker's memory
propagates that mask over the following frames, which is what carries it through frames the
pose model loses to motion blur. It is re-seeded from a fresh prompt every 24 frames - only
on a frame the pose model is sure about - and immediately whenever the propagated mask stops
covering the keypoints, so the memory cannot drift away from the person the way a tracker
prompted once does. There is no text prompt. Holes the mask encloses are filled and islands
far smaller than the body are dropped.

The console reports every step with its duration, how many frames were prompted, propagated,
re-seeded early or left without a prompt, and the mask coverage range - enough to tell from
a log alone what a clip did.

### WanAnimate Preprocess Guard (beta)

Sits between the preprocess and the sampler: takes `mask` and `pose_data`, passes both
through, and stops the workflow with a report when a check of an enabled guard fails,
because sampling on a bad mask or pose is wasted. Every check is normalised by the detected
person's box and crosses the mask with the independently produced pose:

- `pose_guard`: `no_detection`, `pose_low_confidence`, `pose_jump` (torso moving in one
  frame while the box stays), `subject_switch` (the detector picked something else);
  `multi_person` is a warning
- `mask_guard`: `mask_empty`, `mask_leak` (outside the boxes of the frames around it),
  `mask_fragmented` (a second region at least 5% of the main one), `mask_missing_keypoints`
  (confident keypoints outside the mask, named in the report), `mask_unstable` (the mask
  changed, the person did not); `mask_specks` is a warning. The box-based checks only run on
  frames whose box the detector and the pose model agree on - elsewhere an empty or
  out-of-box mask is the pose failing, and the pose checks say so.

`report` is the text that also goes to the console, `metrics` is JSON with every
measurement per frame and `timeline` plots them. Switch both guards off to calibrate the
thresholds on clips you know: nothing stops then and the report, metrics and timeline are
still produced.

## Models

Downloaded on the first run that needs them:

- `yolov10x.onnx` and `rtmw_dw_x_l_wholebody_384x288.onnx` into `ComfyUI/models/detection`,
  from [onnx-community/yolov10x](https://huggingface.co/onnx-community/yolov10x/blob/main/onnx/model.onnx)
  and [bukuroo/RTMW-ONNX](https://huggingface.co/bukuroo/RTMW-ONNX/blob/main/rtmw-l-384.onnx)
  (both saved under the names above), about 340 MB
- `sam3.1_multiplex_fp16.safetensors` into `ComfyUI/models/checkpoints`, from
  [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1), 1.7 GB

The pose model is RTMW-l (MMPose, distilled from RTMW-x, trained on the Cocktail14 mix,
133 COCO-WholeBody keypoints, Apache-2.0). It is OpenMMLab's own ONNX export,
`rtmw-dw-x-l_simcc-cocktail14_270e-384x288`, mirrored on Hugging Face as a plain `.onnx`
instead of their `.zip`; the mirrored file is byte-for-byte the `end2end.onnx` inside that
zip (sha256 `bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054`). Its head is
SimCC rather than a heatmap, decoded in `models/onnx_models.py`.

The ONNX models run through this package's own torch executor: a generic ONNX-op executor
(`models/onnx_graph.py`), with `models/vitpose.py` turning a ViTPose graph into a native
torch module (fused LayerNorm, GELU and scaled-dot-product attention) when one is loaded.
SAM3 is ComfyUI's own implementation.

## Tests

`python -m pytest tests` runs the guard on synthetic clips (no models needed).
`tests/compare_onnxruntime.py` checks the torch executor against onnxruntime on a model
file (install onnxruntime for it).
