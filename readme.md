## ComfyUI helper nodes for [Wan video 2.2 Animate preprocessing](https://github.com/Wan-Video/Wan2.2/tree/main/wan/modules/animate/preprocess)


Nodes to run the ViTPose model, get face crops and keypoint list for SAM2 segmentation.

The ONNX models are run with torch, on whatever device ComfyUI runs on (CUDA, ROCm, MPS
or CPU). There is no onnxruntime dependency, so nothing has to match the CUDA version
torch was built with, and the models take part in ComfyUI's memory management like any
other model: they are moved out of VRAM when another model needs the room and brought
back for the next run.

Models:

to `ComfyUI/models/detection` (subject to change in the future)

YOLO:

https://huggingface.co/onnx-community/yolov10x/blob/main/onnx/model.onnx (save it as `yolov10x.onnx`)

ViTPose ONNX:

The Huge model like in the original code, it's split into two files due to ONNX file size limit:

Both files need to be in same directory, and the onnx file selected in the model loader:

`vitpose_h_wholebody_data.bin` and `vitpose_h_wholebody_model.onnx`

https://huggingface.co/Kijai/vitpose_comfy/tree/main/onnx

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
