"""Make sure onnxruntime in this environment can run on the GPU torch was built for.

ComfyUI-Manager runs this after requirements.txt on install and update. Two things
requirements.txt cannot express:

- 'onnxruntime' (CPU) and 'onnxruntime-gpu' unpack into the same site-packages/onnxruntime
  directory, so whichever any node installs last wins. A CPU wheel pulled in by another
  node's requirements silently removes the CUDA provider.
- onnxruntime-gpu on PyPI is built against CUDA 13 from 1.27 on and CUDA 12 before that,
  so the wheel has to be picked from torch's CUDA major or its CUDA provider fails to load.

Also imported by models/onnx_models.py for the error message when a CUDA session still
comes up on CPU at load time.
"""
import importlib.util
import shutil
import subprocess
import sys

CPU_BUILD = 2
CUDA_LOAD_FAILED = 3

# Runs in a fresh interpreter so it sees the packages as they are on disk right now.
# torch goes first: it loads the CUDA/cuDNN libraries the onnxruntime CUDA provider links against.
_PROBE = r"""
import sys
import torch
import onnxruntime as ort
if "CUDAExecutionProvider" not in ort.get_available_providers():
    sys.exit(%d)
from onnx import helper, TensorProto
x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
model = helper.make_model(helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "probe", [x], [y]),
                          opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
session = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
sys.exit(0 if "CUDAExecutionProvider" in session.get_providers() else %d)
""" % (CPU_BUILD, CUDA_LOAD_FAILED)


def torch_cuda_major():
    """Major CUDA version torch was built with, None for CPU/ROCm/missing torch."""
    try:
        import torch
    except ImportError:
        return None
    if torch.version.cuda is None:
        return None
    return int(torch.version.cuda.split(".")[0])


def onnxruntime_gpu_requirement(cuda_major):
    if cuda_major >= 13:
        return "onnxruntime-gpu>=1.27"
    return "onnxruntime-gpu<1.27"


def cuda_provider_error(requested):
    """Message for a session that was asked for `requested` but came up without it."""
    import onnxruntime

    cuda_major = torch_cuda_major()
    if requested not in onnxruntime.get_available_providers():
        cause = ("the installed 'onnxruntime' package is the CPU build, so the CUDA provider does not "
                 "exist in it (another node's requirements.txt probably replaced onnxruntime-gpu with it)")
    elif cuda_major is None:
        cause = "onnxruntime-gpu is installed but torch has no CUDA build, so there are no CUDA libraries to load"
    else:
        cause = ("the CUDA provider failed to load, almost always onnxruntime-gpu built for a different CUDA major "
                 f"than torch (torch: CUDA {cuda_major}); the onnxruntime log lines above name the missing library")
    spec = onnxruntime_gpu_requirement(cuda_major) if cuda_major is not None else "onnxruntime-gpu"
    return (
        f"{requested} was selected in ONNX Detection Model Loader but the ONNX session came up on CPU: {cause}.\n"
        "Fix with ComfyUI's python and restart:\n"
        f"  {sys.executable} -m pip uninstall -y onnxruntime onnxruntime-gpu\n"
        f"  {sys.executable} -m pip install \"{spec}\"\n"
        "or run install.py in this node's folder, which does the same. "
        "Select CPUExecutionProvider to run on CPU instead."
    )


def probe_cuda_provider():
    result = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
    return result.returncode


def pip(*args):
    if importlib.util.find_spec("pip") is not None:
        cmd = [sys.executable, "-m", "pip", *args]
    elif shutil.which("uv"):
        cmd = ["uv", "pip", *[a for a in args if a != "-y"], "--python", sys.executable]
    else:
        sys.exit(f"[WanAnimatePreprocess] neither pip nor uv is available for {sys.executable}")
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main():
    cuda_major = torch_cuda_major()
    if cuda_major is None:
        print("[WanAnimatePreprocess] torch has no CUDA build, leaving onnxruntime as installed")
        return

    status = probe_cuda_provider()
    if status == 0:
        print("[WanAnimatePreprocess] onnxruntime CUDA provider works")
        return
    if status not in (CPU_BUILD, CUDA_LOAD_FAILED):
        sys.exit("[WanAnimatePreprocess] could not probe onnxruntime, see the error above")

    spec = onnxruntime_gpu_requirement(cuda_major)
    reason = "CPU build of onnxruntime is installed" if status == CPU_BUILD else "CUDA provider failed to load"
    print(f"[WanAnimatePreprocess] {reason}, reinstalling {spec} for torch CUDA {cuda_major}", flush=True)
    pip("uninstall", "-y", "onnxruntime", "onnxruntime-gpu")
    pip("install", spec)

    if probe_cuda_provider() != 0:
        sys.exit(f"[WanAnimatePreprocess] onnxruntime CUDA provider still does not load after installing {spec}")
    print("[WanAnimatePreprocess] onnxruntime CUDA provider works")


if __name__ == "__main__":
    main()
