## ComfyUI helper nodes for [Wan video 2.2 Animate preprocessing](https://github.com/Wan-Video/Wan2.2/tree/main/wan/modules/animate/preprocess)


Nodes to run the ViTPose model, get face crops, keypoints, the person mask and the pose
images for Wan Animate.

`WanAnimate V1 Preprocess` does the whole preprocess in one node: pick the ViTPose, YOLO
and SAM3 models in its widgets (the defaults are downloaded on first use), feed it the
frames and get `pose_images`, `pose_data`, `face_images`, `key_frame_body_points`,
`bboxes`, `face_bboxes` and `mask`. Put `WanAnimate V1 Preprocess Guard (beta)` after it
to stop the workflow when the pose or the mask looks wrong. The older three-node layout
(`ONNX Detection Model Loader` → `Pose and Face Detection` → `Draw ViT Pose`) still works.

The ONNX models are run with torch, on whatever device ComfyUI runs on (CUDA, ROCm, MPS
or CPU). There is no onnxruntime dependency, so nothing has to match the CUDA version
torch was built with, and the models take part in ComfyUI's memory management like any
other model: they are moved out of VRAM when another model needs the room and brought
back for the next run.

Models:

The model widgets list `yolov10x.onnx` and `vitpose_h_wholebody_model.onnx` even before
they exist and download them into `ComfyUI/models/detection` the first time a workflow
runs with them selected (about 2.7 GB in total). To place them yourself, use the same
links and names:

YOLO:

https://huggingface.co/onnx-community/yolov10x/blob/main/onnx/model.onnx (save it as `yolov10x.onnx`)

ViTPose ONNX:

The Huge model like in the original code, it's split into two files due to ONNX file size limit:

Both files need to be in same directory, and the onnx file selected in the model loader:

`vitpose_h_wholebody_data.bin` and `vitpose_h_wholebody_model.onnx`

https://huggingface.co/Kijai/vitpose_comfy/tree/main/onnx

### Person mask (SAM 3 / 3.1)

`WanAnimate V1 Preprocess` (and the older `Pose and Face Detection`) has a `sam3_model`
choice and a `mask` output. With
`sam3.1_multiplex_fp16.safetensors` selected (downloaded into `ComfyUI/models/checkpoints`
on first use, from [Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1)), every
frame's person is segmented by ComfyUI's own SAM3 implementation from that frame's
detected bbox and body keypoints, given to the SAM decoder as a box and positive points.
There is no text prompt and no tracker: frames are independent, so nothing drifts or
ghosts over long clips, and each mask is as good as that frame's detection. Islands far
smaller than the main region are dropped. With `none` the mask output is empty.

### Preprocess guard (beta)

`WanAnimate V1 Preprocess Guard (beta)` sits between the preprocess node and the
sampler: it takes `pose_data` and `mask`, passes both through, and stops the workflow
with a report when the pose or the mask looks wrong, because sampling on a bad mask or
pose is wasted. Every check is normalised by the detected person's size and crosses the
mask with the independently produced pose:

- pose guard: no detection, low keypoint confidence, torso jumping in one frame while
  the box stays, the detector switching to another subject (`subject_switch`); more than
  one confident person is a warning
- mask guard: mask far too small for the box, mask leaking outside the box, a second
  region at least 5% of the main one (a ghost or a second person; smaller specks are
  warnings), confident keypoints outside the mask (a missed hand, foot or limb, named
  in the report), the mask changing between frames while the person does not

`report` is the text that also goes to the console, `metrics` is JSON with every
measurement per frame and `timeline` plots them. Switch both guards off to calibrate the
thresholds on clips you know: nothing stops then and the report, metrics and timeline
are still produced.

### Other model sizes

Any fp32 or fp16 ONNX export of these two model families loads:

- YOLOv10 n / s / m / x, from [onnx-community](https://huggingface.co/onnx-community) (`onnx/model.onnx` or
  `onnx/model_fp16.onnx`) or the [Wan-AI](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/tree/main/process_checkpoint/det)
  yolov10m. Exports with the raw `[1, 84, N]` output (YOLOv8 style) work as well.
- ViTPose wholebody s / b / l / h, single-file exports from
  [JunkyByte/easy_ViTPose](https://huggingface.co/JunkyByte/easy_ViTPose/tree/main/onnx/wholebody) or the split
  Huge model above.

ViTPose graphs are turned into a native torch module (fused LayerNorm, GELU and
scaled-dot-product attention); everything else runs through a generic ONNX-op executor
(`models/onnx_graph.py`) that covers the standard op set of CNN and ViT exports. Quantized
exports (`model_int8`, `model_uint8`, `model_q4`, ...) are not supported. An unsupported
op is reported when the model is loaded, naming the op.


![example](example.png)
