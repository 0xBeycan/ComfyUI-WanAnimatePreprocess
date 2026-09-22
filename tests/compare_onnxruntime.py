"""Check the torch executor against onnxruntime on one ONNX model.

    python tests/compare_onnxruntime.py path/to/model.onnx [--device cuda] [--image frame.png] [--fp16]

Runs the model through this repo's loader (native ViTPose module or the generic graph
executor) and through onnxruntime, prints the max abs difference per output and the time
per run. With --image, a YOLO model (640x640 input) gets the image scaled to 640x640 in
[0, 1] and the confident detections of both runs are printed; a ViTPose model (256x192
input) gets a centre crop with ImageNet normalisation and the heatmap argmax agreement
is printed. Without --image the input is random noise, which is fine for ViTPose but
makes YOLO's TopK pick different low-score rows, so only the tensor diffs mean anything.

onnxruntime is not a dependency of the node; install it (onnxruntime or onnxruntime-gpu)
to run this.
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "models", name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_input(shape, image):
    _, c, h, w = shape
    if image is None:
        return np.random.default_rng(0).random((1, c, h, w), dtype=np.float32)
    import cv2

    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if (h, w) == (640, 640):
        return cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None].copy()
    ih, iw = img.shape[:2]
    cw = min(iw, ih * w // h)
    ch = min(ih, cw * h // w)
    crop = img[(ih - ch) // 2:(ih - ch) // 2 + ch, (iw - cw) // 2:(iw - cw) // 2 + cw]
    crop = cv2.resize(crop, (w, h))
    crop = (crop - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return crop.transpose(2, 0, 1).astype(np.float32)[None].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image")
    parser.add_argument("--fp16", action="store_true", help="run the torch model in float16")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    import onnxruntime as ort

    onnx_graph, vitpose = _load("onnx_graph"), _load("vitpose")
    t0 = time.time()
    graph = onnx_graph.OnnxGraph(args.model)
    net = vitpose.build_vitpose(graph)
    kind = "native ViTPose module" if net is not None else "generic graph executor"
    if net is None:
        net = onnx_graph.GraphModule(graph)
    net = net.eval().to(args.device)
    if args.fp16:
        net = net.half()
    dtype = next(net.parameters()).dtype
    print(f"{os.path.basename(args.model)}: {kind}, opset {graph.opset}, {len(graph.nodes)} nodes, "
          f"built in {time.time() - t0:.1f}s, torch {args.device} {dtype}")

    shape = [d if isinstance(d, int) else 1 for d in graph.input_shapes[graph.inputs[0]]]
    x = prepare_input(shape, args.image)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.device.startswith("cuda") else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.model, providers=providers)
    inp = sess.get_inputs()[0]
    x_ort = x.astype(np.float16) if "float16" in inp.type else x
    ref = sess.run(None, {inp.name: x_ort})
    t0 = time.time()
    for _ in range(args.runs):
        sess.run(None, {inp.name: x_ort})
    t_ort = (time.time() - t0) / args.runs

    def sync():
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif args.device.startswith("mps"):
            torch.mps.synchronize()

    xt = torch.from_numpy(x).to(args.device, dtype)
    with torch.inference_mode():
        out = net(xt)
        out = out if isinstance(out, tuple) else (out,)
        sync()
        t0 = time.time()
        for _ in range(args.runs):
            net(xt)
        sync()
        t_torch = (time.time() - t0) / args.runs
    out = [o.float().cpu().numpy() for o in out]
    print(f"onnxruntime ({sess.get_providers()[0]}): {t_ort * 1000:.1f} ms/run   torch: {t_torch * 1000:.1f} ms/run")

    for name, r, o in zip(graph.outputs, ref, out):
        r = r.astype(np.float32)
        print(f"{name}: shape {tuple(r.shape)}, max |diff| {np.abs(r - o).max():.3e} (values up to {np.abs(r).max():.3e})")
        if r.ndim == 3 and r.shape[-1] == 6:
            for label, arr in (("onnxruntime", r), ("torch", o)):
                rows = arr[0][arr[0][:, 4] > 0.3]
                print(f"  {label}: {len(rows)} detections with score > 0.3: {np.round(rows, 2).tolist()}")
        if r.ndim == 4 and r.shape[1] > 1 and args.image:
            same = (r[0].reshape(r.shape[1], -1).argmax(1) == o[0].reshape(o.shape[1], -1).argmax(1)).sum()
            print(f"  heatmap argmax agreement: {same} / {r.shape[1]}")


if __name__ == "__main__":
    sys.exit(main())
