"""Run an ONNX graph with torch.

The detection models ship as ONNX files. Instead of onnxruntime, whose GPU wheel is built
for one CUDA major and has to match the one torch was built with, the graph is executed
node by node with torch ops on whatever device ComfyUI runs on. Weights are read straight
from the .onnx file (and its external data file(s), if any), so the model downloads stay
the same.

Only the standard ONNX op set used by CNN / ViT exports is implemented; an unsupported op
is reported when the model is loaded, not in the middle of a run.
"""
import math
import os

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnx import TensorProto, numpy_helper

_DTYPES = {
    TensorProto.FLOAT: torch.float32,
    TensorProto.FLOAT16: torch.float16,
    TensorProto.BFLOAT16: torch.bfloat16,
    TensorProto.DOUBLE: torch.float64,
    TensorProto.INT8: torch.int8,
    TensorProto.INT16: torch.int16,
    TensorProto.INT32: torch.int32,
    TensorProto.INT64: torch.int64,
    TensorProto.UINT8: torch.uint8,
    TensorProto.BOOL: torch.bool,
}


class Node:
    __slots__ = ("op", "name", "inputs", "outputs", "attrs", "consts")

    def __init__(self, proto):
        self.op = proto.op_type
        self.name = proto.name
        self.inputs = list(proto.input)
        self.outputs = list(proto.output)
        self.attrs = {a.name: _attribute(a) for a in proto.attribute}
        # python values of small constant inputs, filled in by GraphModule
        self.consts = {}


def _attribute(a):
    v = onnx.helper.get_attribute_value(a)
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, TensorProto):
        return torch.from_numpy(numpy_helper.to_array(v).copy())
    if isinstance(v, list) and v and isinstance(v[0], bytes):
        return [s.decode() for s in v]
    return v


def _read_tensor(t, base_dir):
    """Initializer -> numpy, reading external data straight from its file."""
    if t.data_location == TensorProto.EXTERNAL:
        info = {e.key: e.value for e in t.external_data}
        dtype = onnx.helper.tensor_dtype_to_np_dtype(t.data_type)
        count = int(np.prod(t.dims)) if len(t.dims) else 1
        path = os.path.join(base_dir, info["location"])
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{os.path.basename(base_dir)}: the ONNX model stores its weights in '{info['location']}', "
                f"which is missing next to the .onnx file (looked in {base_dir})")
        arr = np.fromfile(path, dtype=dtype, count=count, offset=int(info.get("offset", 0)))
        if arr.size != count:
            raise ValueError(f"{path}: expected {count} values for {t.name}, read {arr.size}")
        return arr.reshape(tuple(t.dims))
    return numpy_helper.to_array(t)


class OnnxGraph:
    """A parsed ONNX model: constant tensors (CPU torch tensors), nodes and the I/O names."""

    def __init__(self, path):
        model = onnx.load(path, load_external_data=False)
        base_dir = os.path.dirname(os.path.abspath(path))
        self.path = path
        self.opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 13)
        self.tensors = {}
        for t in model.graph.initializer:
            arr = _read_tensor(t, base_dir)
            if not arr.flags.writeable:  # in-file raw data comes back as a read-only view
                arr = arr.copy()
            self.tensors[t.name] = torch.from_numpy(np.ascontiguousarray(arr))
        self.nodes = []
        for n in model.graph.node:
            if n.op_type == "Constant":
                self.tensors[n.output[0]] = _constant_value(n)
            else:
                self.nodes.append(Node(n))
        self.inputs = [i.name for i in model.graph.input if i.name not in self.tensors]
        self.outputs = [o.name for o in model.graph.output]
        self.input_shapes = {
            i.name: [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim]
            for i in model.graph.input if i.name in self.inputs
        }
        self.input_dtypes = {}
        for i in model.graph.input:
            if i.name in self.inputs:
                if i.type.tensor_type.elem_type not in _DTYPES:
                    raise ValueError(f"{os.path.basename(path)}: input {i.name} has an unsupported tensor type "
                                     f"({i.type.tensor_type.elem_type})")
                self.input_dtypes[i.name] = _DTYPES[i.type.tensor_type.elem_type]

    def producer(self, name):
        for n in self.nodes:
            if name in n.outputs:
                return n
        return None

    def consumers(self, name):
        return [n for n in self.nodes if name in n.inputs]


def _constant_value(n):
    attrs = {a.name: _attribute(a) for a in n.attribute}
    if "value" in attrs:
        return attrs["value"]
    for key in ("value_float", "value_int"):
        if key in attrs:
            return torch.tensor(attrs[key])
    for key in ("value_floats", "value_ints"):
        if key in attrs:
            return torch.tensor(attrs[key])
    raise ValueError(f"Constant node {n.name}: unsupported value attribute {list(attrs)}")


def _pairs_to_torch_pad(pads):
    """ONNX [b0, b1, .., e0, e1, ..] -> torch (b_last, e_last, .., b0, e0)."""
    half = len(pads) // 2
    out = []
    for i in reversed(range(half)):
        out += [pads[i], pads[half + i]]
    return out


def _same_pad(size, kernel, stride, dilation, lower):
    out = math.ceil(size / stride)
    total = max((out - 1) * stride + (kernel - 1) * dilation + 1 - size, 0)
    small, big = total // 2, total - total // 2
    return (big, small) if lower else (small, big)


QUANTIZED_OPS = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "QLinearConv", "QLinearMatMul",
                 "MatMulInteger", "ConvInteger"}
RESIZE_MODES = {("nearest", "asymmetric", "floor"), ("nearest", "half_pixel", "round_prefer_floor"),
                ("nearest", "half_pixel", "round_prefer_ceil"), ("nearest", "pytorch_half_pixel", "round_prefer_floor"),
                ("nearest", "pytorch_half_pixel", "round_prefer_ceil")}
LINEAR_COORDS = {"half_pixel", "pytorch_half_pixel", "align_corners"}


class GraphModule(nn.Module):
    """Executes an OnnxGraph. Float constants are parameters (they move with the module);
    integer/bool constants are shape and index arithmetic and stay on the CPU."""

    def __init__(self, graph):
        super().__init__()
        self.opset = graph.opset
        self.input_names = graph.inputs
        self.output_names = graph.outputs
        self.params = nn.ParameterDict()
        self.consts = {}
        self._param_key = {}
        for i, (name, t) in enumerate(graph.tensors.items()):
            if t.is_floating_point():
                key = f"t{i}"
                self.params[key] = nn.Parameter(t, requires_grad=False)
                self._param_key[name] = key
            else:
                self.consts[name] = t

        name = os.path.basename(graph.path)
        quantized = sorted({n.op for n in graph.nodes if n.op in QUANTIZED_OPS})
        if quantized:
            raise ValueError(f"{name} is a quantized ONNX export ({', '.join(quantized)}); use the fp32 or fp16 model")
        missing = sorted({n.op for n in graph.nodes if not hasattr(self, "op_" + n.op)})
        if missing:
            raise ValueError(f"{name} uses ONNX ops this loader does not implement: {', '.join(missing)}")
        for n in graph.nodes:
            problem = self._unsupported_attributes(n)
            if problem:
                raise ValueError(f"{name}: {n.op} ({n.name}) {problem}")
        self.nodes = graph.nodes
        self._ops = [getattr(self, "op_" + n.op) for n in self.nodes]

        # small constant inputs as python values, resolved once instead of per run
        for n in self.nodes:
            for i, name in enumerate(n.inputs):
                t = graph.tensors.get(name)
                if t is not None and t.numel() <= 64:
                    n.consts[i] = t.tolist()

        # free intermediates after their last consumer so activations do not pile up
        last_use = {}
        for i, n in enumerate(self.nodes):
            for name in n.inputs:
                if name and name not in graph.tensors:
                    last_use[name] = i
        keep = set(self.output_names)
        self._release = [[] for _ in self.nodes]
        for name, i in last_use.items():
            if name not in keep:
                self._release[i].append(name)

    def forward(self, *args):
        env = dict(self.consts)
        for name, key in self._param_key.items():
            env[name] = self.params[key]
        for name, x in zip(self.input_names, args):
            env[name] = x
        for i, (node, op) in enumerate(zip(self.nodes, self._ops)):
            ins = [env[name] if name else None for name in node.inputs]
            out = op(node, *ins)
            if not isinstance(out, tuple):
                out = (out,)
            for name, val in zip(node.outputs, out):
                if name:
                    env[name] = val
            for name in self._release[i]:
                env.pop(name, None)
        outs = tuple(env[name] for name in self.output_names)
        return outs[0] if len(outs) == 1 else outs

    # -- helpers ---------------------------------------------------------------------------

    def _unsupported_attributes(self, node):
        """Why this node's attributes cannot be run, or None. Checked when the model is
        loaded so a run never fails on a variant the ops do not handle."""
        a = node.attrs
        if node.op == "Cast" and a.get("to") not in _DTYPES:
            return f"casts to an unsupported tensor type ({a.get('to')})"
        if node.op == "Resize":
            mode = a.get("mode", "nearest")
            coord = a.get("coordinate_transformation_mode", "half_pixel")
            if self.opset < 11:
                # opset 10 Resize has no coordinate attributes: nearest is asymmetric / floor
                return None if mode == "nearest" else f"resizes with {mode} at opset 10, which is not supported"
            if "axes" in a:
                return "uses the axes attribute, which is not supported"
            if mode == "nearest" and (mode, coord, a.get("nearest_mode", "round_prefer_floor")) not in RESIZE_MODES:
                return f"resizes with nearest/{coord}/{a.get('nearest_mode', 'round_prefer_floor')}, which is not supported"
            if mode in ("linear", "cubic") and coord not in LINEAR_COORDS:
                return f"resizes with {mode}/{coord}, which is not supported"
            if mode == "cubic" and a.get("cubic_coeff_a", -0.75) != -0.75:
                return "uses a cubic_coeff_a other than -0.75, which is not supported"
            if mode not in ("nearest", "linear", "cubic"):
                return f"resizes with mode {mode}, which is not supported"
        if node.op == "Upsample" and a.get("mode", "nearest") != "nearest":
            return f"upsamples with mode {a.get('mode')}, which is not supported (nearest only)"
        if node.op == "Pad" and a.get("mode", "constant") not in ("constant", "reflect", "edge", "wrap"):
            return f"pads with mode {a.get('mode')}, which is not supported"
        return None

    @staticmethod
    def _value(node, idx, tensor):
        if idx in node.consts:
            return node.consts[idx]
        return tensor.tolist()

    @staticmethod
    def _ints(node, idx, tensor):
        v = GraphModule._value(node, idx, tensor)
        return [int(v)] if not isinstance(v, list) else [int(x) for x in v]

    @staticmethod
    def _together(*tensors):
        """Move CPU operands next to a device operand; shape arithmetic stays on the CPU."""
        device = None
        for t in tensors:
            if t is not None and t.device.type != "cpu":
                device = t.device
                break
        if device is None:
            return tensors
        return tuple(t.to(device) if t is not None and t.device != device else t for t in tensors)

    def _binary(self, a, b, fn):
        a, b = self._together(a, b)
        return fn(a, b)

    def _axes(self, node, idx, tensor, attr="axes"):
        if idx < len(node.inputs) and node.inputs[idx]:
            return self._ints(node, idx, tensor)
        return node.attrs.get(attr)

    # -- elementwise -----------------------------------------------------------------------

    def op_Add(self, node, a, b):
        return self._binary(a, b, torch.add)

    def op_Sub(self, node, a, b):
        return self._binary(a, b, torch.sub)

    def op_Mul(self, node, a, b):
        return self._binary(a, b, torch.mul)

    def op_Div(self, node, a, b):
        a, b = self._together(a, b)
        if not a.is_floating_point() and not b.is_floating_point():
            return torch.div(a, b, rounding_mode="trunc")
        return torch.div(a, b)

    def op_Pow(self, node, a, b):
        a, b = self._together(a, b)
        return torch.pow(a, b.to(a.dtype) if a.is_floating_point() else b)

    def op_Mod(self, node, a, b):
        a, b = self._together(a, b)
        return torch.fmod(a, b) if node.attrs.get("fmod", 0) else torch.remainder(a, b)

    def op_Max(self, node, *xs):
        xs = self._together(*xs)
        out = xs[0]
        for x in xs[1:]:
            out = torch.maximum(out, x)
        return out

    def op_Min(self, node, *xs):
        xs = self._together(*xs)
        out = xs[0]
        for x in xs[1:]:
            out = torch.minimum(out, x)
        return out

    def op_Sum(self, node, *xs):
        xs = self._together(*xs)
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out

    def op_Mean(self, node, *xs):
        return self.op_Sum(node, *xs) / len(xs)

    def op_Neg(self, node, x):
        return -x

    def op_Abs(self, node, x):
        return x.abs()

    def op_Floor(self, node, x):
        return x.floor()

    def op_Ceil(self, node, x):
        return x.ceil()

    def op_Round(self, node, x):
        return x.round()

    def op_Sign(self, node, x):
        return x.sign()

    def op_Sqrt(self, node, x):
        return x.sqrt()

    def op_Exp(self, node, x):
        return x.exp()

    def op_Log(self, node, x):
        return x.log()

    def op_Sin(self, node, x):
        return x.sin()

    def op_Cos(self, node, x):
        return x.cos()

    def op_Tan(self, node, x):
        return x.tan()

    def op_Atan(self, node, x):
        return x.atan()

    def op_Erf(self, node, x):
        return x.erf()

    def op_Reciprocal(self, node, x):
        return x.reciprocal()

    def op_Sigmoid(self, node, x):
        return x.sigmoid()

    def op_Tanh(self, node, x):
        return x.tanh()

    def op_Relu(self, node, x):
        return F.relu(x)

    def op_LeakyRelu(self, node, x):
        return F.leaky_relu(x, node.attrs.get("alpha", 0.01))

    def op_PRelu(self, node, x, slope):
        x, slope = self._together(x, slope)
        return torch.where(x >= 0, x, x * slope)

    def op_Elu(self, node, x):
        return F.elu(x, node.attrs.get("alpha", 1.0))

    def op_Selu(self, node, x):
        return F.selu(x)

    def op_Softplus(self, node, x):
        return F.softplus(x)

    def op_Softsign(self, node, x):
        return F.softsign(x)

    def op_HardSigmoid(self, node, x):
        return torch.clamp(node.attrs.get("alpha", 0.2) * x + node.attrs.get("beta", 0.5), 0, 1)

    def op_HardSwish(self, node, x):
        return F.hardswish(x)

    def op_Mish(self, node, x):
        return F.mish(x)

    def op_Gelu(self, node, x):
        return F.gelu(x, approximate=node.attrs.get("approximate", "none"))

    def op_Clip(self, node, x, lo=None, hi=None):
        if self.opset < 11:
            lo, hi = node.attrs.get("min"), node.attrs.get("max")
        else:
            lo = None if lo is None else self._value(node, 1, lo)
            hi = None if hi is None else self._value(node, 2, hi)
        return torch.clamp(x, lo, hi)

    def op_Identity(self, node, x):
        return x

    def op_Dropout(self, node, x, *rest):
        if len(node.outputs) > 1 and node.outputs[1]:
            return x, torch.ones_like(x, dtype=torch.bool)
        return x

    def op_Not(self, node, x):
        return ~x

    def op_And(self, node, a, b):
        return self._binary(a, b, torch.logical_and)

    def op_Or(self, node, a, b):
        return self._binary(a, b, torch.logical_or)

    def op_Xor(self, node, a, b):
        return self._binary(a, b, torch.logical_xor)

    def op_Equal(self, node, a, b):
        return self._binary(a, b, torch.eq)

    def op_Greater(self, node, a, b):
        return self._binary(a, b, torch.gt)

    def op_GreaterOrEqual(self, node, a, b):
        return self._binary(a, b, torch.ge)

    def op_Less(self, node, a, b):
        return self._binary(a, b, torch.lt)

    def op_LessOrEqual(self, node, a, b):
        return self._binary(a, b, torch.le)

    def op_Where(self, node, cond, a, b):
        cond, a, b = self._together(cond, a, b)
        return torch.where(cond, a, b)

    def op_IsNaN(self, node, x):
        return torch.isnan(x)

    def op_IsInf(self, node, x):
        return torch.isinf(x)

    def op_Cast(self, node, x):
        return x.to(_DTYPES[node.attrs["to"]])

    def op_CastLike(self, node, x, like):
        return x.to(like.dtype)

    # -- matmul ----------------------------------------------------------------------------

    def op_MatMul(self, node, a, b):
        return self._binary(a, b, torch.matmul)

    def op_Gemm(self, node, a, b, c=None):
        a, b, c = self._together(a, b, c)
        if node.attrs.get("transA", 0):
            a = a.t()
        if node.attrs.get("transB", 0):
            b = b.t()
        out = node.attrs.get("alpha", 1.0) * (a @ b)
        if c is not None:
            out = out + node.attrs.get("beta", 1.0) * c
        return out

    def op_Einsum(self, node, *xs):
        return torch.einsum(node.attrs["equation"], *self._together(*xs))

    # -- shape -----------------------------------------------------------------------------

    def op_Shape(self, node, x):
        shape = list(x.shape)
        start = node.attrs.get("start", 0)
        end = node.attrs.get("end", len(shape))
        return torch.tensor(shape[start:end], dtype=torch.int64)

    def op_Size(self, node, x):
        return torch.tensor(x.numel(), dtype=torch.int64)

    def op_Reshape(self, node, x, shape):
        shape = self._ints(node, 1, shape)
        if not node.attrs.get("allowzero", 0):
            shape = [x.shape[i] if s == 0 else s for i, s in enumerate(shape)]
        return x.reshape(shape)

    def op_Transpose(self, node, x):
        perm = node.attrs.get("perm") or list(reversed(range(x.dim())))
        return x.permute(perm)

    def op_Flatten(self, node, x):
        axis = node.attrs.get("axis", 1)
        return x.reshape(1, -1) if axis == 0 else x.reshape(math.prod(x.shape[:axis]), -1)

    def op_Squeeze(self, node, x, axes=None):
        axes = self._axes(node, 1, axes)
        if not axes:
            return x.squeeze()
        for a in sorted((a + x.dim() if a < 0 else a for a in axes), reverse=True):
            x = x.squeeze(a)
        return x

    def op_Unsqueeze(self, node, x, axes=None):
        axes = self._axes(node, 1, axes)
        rank = x.dim() + len(axes)
        for a in sorted(a + rank if a < 0 else a for a in axes):
            x = x.unsqueeze(a)
        return x

    def op_Concat(self, node, *xs):
        return torch.cat(self._together(*xs), dim=node.attrs.get("axis", 0))

    def op_Split(self, node, x, split=None):
        axis = node.attrs.get("axis", 0)
        if split is not None:
            sizes = self._ints(node, 1, split)
        elif "split" in node.attrs:
            sizes = node.attrs["split"]
        else:
            n = node.attrs.get("num_outputs", len(node.outputs))
            size = math.ceil(x.shape[axis] / n)
            sizes = [size] * (n - 1) + [x.shape[axis] - size * (n - 1)]
        return tuple(torch.split(x, sizes, dim=axis))

    def op_Slice(self, node, x, starts=None, ends=None, axes=None, steps=None):
        if self.opset < 10:
            starts, ends, axes = node.attrs["starts"], node.attrs["ends"], node.attrs.get("axes")
            steps = None
        else:
            starts = self._ints(node, 1, starts)
            ends = self._ints(node, 2, ends)
            axes = None if axes is None else self._ints(node, 3, axes)
            steps = None if steps is None else self._ints(node, 4, steps)
        axes = axes or list(range(len(starts)))
        steps = steps or [1] * len(starts)
        index = [slice(None)] * x.dim()
        for start, end, axis, step in zip(starts, ends, axes, steps):
            axis = axis + x.dim() if axis < 0 else axis
            dim = x.shape[axis]
            if step > 0:
                start = min(max(start + dim if start < 0 else start, 0), dim)
                end = min(max(end + dim if end < 0 else end, 0), dim)
                index[axis] = slice(start, end, step)
            else:
                start = min(max(start + dim if start < 0 else start, 0), dim - 1)
                end = max(min(end + dim if end < 0 else end, dim - 1), -1)
                idx = torch.arange(start, end, step, device=x.device)
                x = x.index_select(axis, idx)
        return x[tuple(index)]

    def op_Gather(self, node, x, idx):
        axis = node.attrs.get("axis", 0)
        axis = axis + x.dim() if axis < 0 else axis
        x, idx = self._together(x, idx)
        idx = idx.long()
        idx = torch.where(idx < 0, idx + x.shape[axis], idx)
        out = x.index_select(axis, idx.reshape(-1))
        return out.reshape(x.shape[:axis] + idx.shape + x.shape[axis + 1:])

    def op_GatherElements(self, node, x, idx):
        axis = node.attrs.get("axis", 0)
        x, idx = self._together(x, idx)
        idx = idx.long()
        idx = torch.where(idx < 0, idx + x.shape[axis], idx)
        return torch.gather(x, axis, idx)

    def op_GatherND(self, node, x, idx):
        batch_dims = node.attrs.get("batch_dims", 0)
        x, idx = self._together(x, idx)
        idx = idx.long()
        if batch_dims:
            batch_shape = x.shape[:batch_dims]
            x = x.reshape(-1, *x.shape[batch_dims:])
            idx = idx.reshape(-1, *idx.shape[batch_dims:])
            b = torch.arange(idx.shape[0], device=idx.device).reshape(-1, *([1] * (idx.dim() - 2)), 1)
            b = b.expand(*idx.shape[:-1], 1)
            idx = torch.cat([b, idx], dim=-1)
            out = x[tuple(idx.movedim(-1, 0))]
            return out.reshape(*batch_shape, *out.shape[1:])
        return x[tuple(idx.movedim(-1, 0))]

    def op_ScatterElements(self, node, x, idx, updates):
        x, idx, updates = self._together(x, idx, updates)
        reduction = node.attrs.get("reduction", "none")
        if reduction == "none":
            return x.clone().scatter_(node.attrs.get("axis", 0), idx.long(), updates)
        return x.clone().scatter_reduce_(node.attrs.get("axis", 0), idx.long(), updates,
                                         {"add": "sum", "mul": "prod", "max": "amax", "min": "amin"}[reduction])

    def op_ScatterND(self, node, x, idx, updates):
        x, idx, updates = self._together(x, idx, updates)
        out = x.clone()
        out[tuple(idx.long().movedim(-1, 0))] = updates
        return out

    def op_Expand(self, node, x, shape):
        return x.expand(torch.broadcast_shapes(x.shape, tuple(self._ints(node, 1, shape))))

    def op_Tile(self, node, x, repeats):
        return x.repeat(self._ints(node, 1, repeats))

    def op_ConstantOfShape(self, node, shape):
        value = node.attrs.get("value")
        if value is None:
            value = torch.tensor(0.0)
        return torch.full(self._ints(node, 0, shape), value.item(), dtype=value.dtype)

    def op_Range(self, node, start, limit, delta):
        s, l, d = self._value(node, 0, start), self._value(node, 1, limit), self._value(node, 2, delta)
        return torch.arange(s, l, d, dtype=start.dtype)

    def op_NonZero(self, node, x):
        return torch.nonzero(x).t()

    def op_Pad(self, node, x, pads=None, value=None, axes=None):
        mode = node.attrs.get("mode", "constant")
        if self.opset < 11:
            pads = node.attrs["pads"]
            value = node.attrs.get("value", 0.0)
        else:
            pads = self._ints(node, 1, pads)
            value = 0.0 if value is None else self._value(node, 2, value)
            if axes is not None:
                axes = self._ints(node, 3, axes)
                full = [0] * (2 * x.dim())
                for i, a in enumerate(axes):
                    a = a + x.dim() if a < 0 else a
                    full[a], full[x.dim() + a] = pads[i], pads[len(axes) + i]
                pads = full
        torch_pads = _pairs_to_torch_pad(pads)
        if mode == "constant":
            return F.pad(x, torch_pads, value=value)
        # reflect / edge / wrap only pad the trailing (spatial) dims in torch
        while len(torch_pads) > 2 and torch_pads[-1] == 0 and torch_pads[-2] == 0:
            torch_pads = torch_pads[:-2]
        return F.pad(x, torch_pads, mode={"reflect": "reflect", "edge": "replicate", "wrap": "circular"}[mode])

    # -- reductions ------------------------------------------------------------------------

    def _reduce(self, node, x, axes, fn):
        axes = self._axes(node, 1, axes)
        keepdim = bool(node.attrs.get("keepdims", 1))
        if not axes:
            if node.attrs.get("noop_with_empty_axes", 0):
                return x
            axes = list(range(x.dim()))
        return fn(x, axes, keepdim)

    def op_ReduceMean(self, node, x, axes=None):
        return self._reduce(node, x, axes, lambda t, d, k: t.mean(d, keepdim=k))

    def op_ReduceSum(self, node, x, axes=None):
        return self._reduce(node, x, axes, lambda t, d, k: t.sum(d, keepdim=k))

    def op_ReduceMax(self, node, x, axes=None):
        return self._reduce(node, x, axes, lambda t, d, k: t.amax(d, keepdim=k))

    def op_ReduceMin(self, node, x, axes=None):
        return self._reduce(node, x, axes, lambda t, d, k: t.amin(d, keepdim=k))

    def op_ReduceProd(self, node, x, axes=None):
        def prod(t, dims, k):
            for d in sorted((d + t.dim() if d < 0 else d for d in dims), reverse=True):
                t = t.prod(d, keepdim=k)
            return t
        return self._reduce(node, x, axes, prod)

    def op_ReduceL1(self, node, x, axes=None):
        return self._reduce(node, x, axes, lambda t, d, k: t.abs().sum(d, keepdim=k))

    def op_ReduceL2(self, node, x, axes=None):
        return self._reduce(node, x, axes, lambda t, d, k: t.pow(2).sum(d, keepdim=k).sqrt())

    def op_ReduceSumSquare(self, node, x, axes=None):
        return self._reduce(node, x, axes, lambda t, d, k: t.pow(2).sum(d, keepdim=k))

    def op_ReduceLogSum(self, node, x, axes=None):
        return self._reduce(node, x, axes, lambda t, d, k: t.sum(d, keepdim=k).log())

    def op_ReduceLogSumExp(self, node, x, axes=None):
        return self._reduce(node, x, axes, lambda t, d, k: t.logsumexp(d, keepdim=k))

    def _arg(self, node, x, fn):
        axis = node.attrs.get("axis", 0)
        keepdim = bool(node.attrs.get("keepdims", 1))
        if node.attrs.get("select_last_index", 0):
            out = fn(x.flip(axis), dim=axis, keepdim=keepdim)
            return x.shape[axis] - 1 - out
        return fn(x, dim=axis, keepdim=keepdim)

    def op_ArgMax(self, node, x):
        return self._arg(node, x, torch.argmax)

    def op_ArgMin(self, node, x):
        return self._arg(node, x, torch.argmin)

    def op_TopK(self, node, x, k):
        k = self._ints(node, 1, k)[0]
        values, indices = torch.topk(x, k, dim=node.attrs.get("axis", -1),
                                     largest=bool(node.attrs.get("largest", 1)),
                                     sorted=bool(node.attrs.get("sorted", 1)))
        return values, indices

    def op_CumSum(self, node, x, axis):
        axis = self._ints(node, 1, axis)[0]
        if node.attrs.get("reverse", 0):
            x = x.flip(axis)
        out = x.cumsum(axis)
        if node.attrs.get("exclusive", 0):
            out = out - x
        return out.flip(axis) if node.attrs.get("reverse", 0) else out

    def _softmax(self, node, x, fn):
        if self.opset >= 13:
            return fn(x, dim=node.attrs.get("axis", -1))
        # older opsets flatten to 2D at `axis` and normalise over the trailing block
        axis = node.attrs.get("axis", 1)
        axis = axis + x.dim() if axis < 0 else axis
        return fn(x.reshape(math.prod(x.shape[:axis]), -1), dim=-1).reshape(x.shape)

    def op_Softmax(self, node, x):
        return self._softmax(node, x, F.softmax)

    def op_LogSoftmax(self, node, x):
        return self._softmax(node, x, F.log_softmax)

    def op_NonMaxSuppression(self, node, boxes, scores, max_out=None, iou=None, score_thr=None):
        from torchvision.ops import nms

        max_out = 0 if max_out is None else self._ints(node, 2, max_out)[0]
        iou = 0.0 if iou is None else float(self._value(node, 3, iou))
        score_thr = None if score_thr is None else float(self._value(node, 4, score_thr))
        boxes, scores = self._together(boxes, scores)
        if node.attrs.get("center_point_box", 0):
            cx, cy, w, h = boxes.unbind(-1)
            boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
        selected = []
        for b in range(scores.shape[0]):
            for c in range(scores.shape[1]):
                s = scores[b, c]
                idx = torch.arange(s.shape[0], device=s.device)
                if score_thr is not None:
                    idx = idx[s > score_thr]
                keep = idx[nms(boxes[b, idx].float(), s[idx].float(), iou)]
                if max_out:
                    keep = keep[:max_out]
                bc = torch.tensor([b, c], device=keep.device).expand(keep.shape[0], 2)
                selected.append(torch.cat([bc, keep[:, None]], dim=1))
        return torch.cat(selected, dim=0) if selected else torch.zeros(0, 3, dtype=torch.int64)

    # -- neural-net layers -----------------------------------------------------------------

    def _conv_pads(self, node, x, kernel, strides, dilations):
        rank = len(kernel)
        auto = node.attrs.get("auto_pad", "NOTSET")
        if auto in ("SAME_UPPER", "SAME_LOWER"):
            pairs = [_same_pad(x.shape[2 + i], kernel[i], strides[i], dilations[i], auto == "SAME_LOWER")
                     for i in range(rank)]
            return [p[0] for p in pairs] + [p[1] for p in pairs]
        if auto == "VALID":
            return [0] * (2 * rank)
        return node.attrs.get("pads", [0] * (2 * rank))

    def op_Conv(self, node, x, w, b=None):
        x, w, b = self._together(x, w, b)
        rank = x.dim() - 2
        kernel = node.attrs.get("kernel_shape", list(w.shape[2:]))
        strides = node.attrs.get("strides", [1] * rank)
        dilations = node.attrs.get("dilations", [1] * rank)
        pads = self._conv_pads(node, x, kernel, strides, dilations)
        begin, end = pads[:rank], pads[rank:]
        if begin == end:
            padding = begin
        else:
            x = F.pad(x, _pairs_to_torch_pad(pads))
            padding = 0
        conv = (F.conv1d, F.conv2d, F.conv3d)[rank - 1]
        return conv(x, w, b, stride=strides, padding=padding, dilation=dilations, groups=node.attrs.get("group", 1))

    def op_ConvTranspose(self, node, x, w, b=None):
        x, w, b = self._together(x, w, b)
        rank = x.dim() - 2
        kernel = node.attrs.get("kernel_shape", list(w.shape[2:]))
        strides = node.attrs.get("strides", [1] * rank)
        dilations = node.attrs.get("dilations", [1] * rank)
        output_padding = node.attrs.get("output_padding", [0] * rank)
        groups = node.attrs.get("group", 1)
        conv = (F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d)[rank - 1]
        if "output_shape" in node.attrs:
            full = [(x.shape[2 + i] - 1) * strides[i] + output_padding[i] + (kernel[i] - 1) * dilations[i] + 1
                    for i in range(rank)]
            total = [full[i] - node.attrs["output_shape"][i] for i in range(rank)]
            lower = node.attrs.get("auto_pad", "NOTSET") == "SAME_LOWER"
            pads = [t - t // 2 if lower else t // 2 for t in total] + [t // 2 if lower else t - t // 2 for t in total]
        else:
            pads = node.attrs.get("pads", [0] * (2 * rank))
        begin, end = pads[:rank], pads[rank:]
        if begin == end:
            return conv(x, w, b, stride=strides, padding=begin, output_padding=output_padding,
                        groups=groups, dilation=dilations)
        out = conv(x, w, b, stride=strides, padding=0, output_padding=output_padding, groups=groups, dilation=dilations)
        index = [slice(None), slice(None)] + [slice(begin[i], out.shape[2 + i] - end[i]) for i in range(rank)]
        return out[tuple(index)]

    def op_BatchNormalization(self, node, x, scale, bias, mean, var):
        x, scale, bias, mean, var = self._together(x, scale, bias, mean, var)
        return F.batch_norm(x, mean, var, weight=scale, bias=bias, training=False, eps=node.attrs.get("epsilon", 1e-5))

    def op_InstanceNormalization(self, node, x, scale, bias):
        x, scale, bias = self._together(x, scale, bias)
        return F.instance_norm(x, weight=scale, bias=bias, eps=node.attrs.get("epsilon", 1e-5))

    def op_LayerNormalization(self, node, x, scale, bias=None):
        x, scale, bias = self._together(x, scale, bias)
        axis = node.attrs.get("axis", -1)
        axis = axis + x.dim() if axis < 0 else axis
        eps = node.attrs.get("epsilon", 1e-5)
        out = F.layer_norm(x, x.shape[axis:], weight=scale, bias=bias, eps=eps)
        if len(node.outputs) > 1 and any(node.outputs[1:]):
            dims = list(range(axis, x.dim()))
            mean = x.mean(dims, keepdim=True)
            inv_std = torch.rsqrt(x.var(dims, unbiased=False, keepdim=True) + eps)
            return out, mean, inv_std
        return out

    def op_GroupNormalization(self, node, x, scale, bias):
        x, scale, bias = self._together(x, scale, bias)
        return F.group_norm(x, node.attrs["num_groups"], weight=scale, bias=bias, eps=node.attrs.get("epsilon", 1e-5))

    def _pool(self, node, x, fn, pad_value):
        rank = x.dim() - 2
        kernel = node.attrs["kernel_shape"]
        strides = node.attrs.get("strides", [1] * rank)
        dilations = node.attrs.get("dilations", [1] * rank)
        pads = self._conv_pads(node, x, kernel, strides, dilations)
        begin, end = pads[:rank], pads[rank:]
        if begin != end:
            x = F.pad(x, _pairs_to_torch_pad(pads), value=pad_value)
            begin = [0] * rank
        return fn(x, kernel, strides, begin, dilations, bool(node.attrs.get("ceil_mode", 0)))

    def op_MaxPool(self, node, x):
        rank = x.dim() - 2
        pool = (F.max_pool1d, F.max_pool2d, F.max_pool3d)[rank - 1]
        return self._pool(node, x, lambda t, k, s, p, d, c: pool(t, k, s, p, d, ceil_mode=c), float("-inf"))

    def op_AveragePool(self, node, x):
        rank = x.dim() - 2
        pool = (F.avg_pool1d, F.avg_pool2d, F.avg_pool3d)[rank - 1]
        include_pad = bool(node.attrs.get("count_include_pad", 0))
        pads = node.attrs.get("pads", [0] * (2 * rank))
        if pads[:rank] != pads[rank:] and not include_pad:
            raise ValueError(f"AveragePool {node.name}: asymmetric pads with count_include_pad=0 are not supported")
        return self._pool(node, x, lambda t, k, s, p, d, c: pool(t, k, s, p, ceil_mode=c, count_include_pad=include_pad), 0.0)

    def op_GlobalAveragePool(self, node, x):
        return x.mean(list(range(2, x.dim())), keepdim=True)

    def op_GlobalMaxPool(self, node, x):
        return x.amax(list(range(2, x.dim())), keepdim=True)

    def _interpolate(self, node, x, mode, coord, nearest_mode, scales, sizes, antialias):
        rank = x.dim() - 2
        if mode == "nearest":
            if coord == "asymmetric" and nearest_mode == "floor":
                torch_mode, align = "nearest", None
            elif coord in ("half_pixel", "pytorch_half_pixel") and nearest_mode in ("round_prefer_floor", "round_prefer_ceil"):
                torch_mode, align = "nearest-exact", None
            else:
                raise ValueError(f"Resize {node.name}: nearest with {coord}/{nearest_mode} is not supported")
        elif mode in ("linear", "cubic"):
            if coord in ("half_pixel", "pytorch_half_pixel"):
                align = False
            elif coord == "align_corners":
                align = True
            else:
                raise ValueError(f"Resize {node.name}: {mode} with {coord} is not supported")
            if mode == "cubic" and node.attrs.get("cubic_coeff_a", -0.75) != -0.75:
                raise ValueError(f"Resize {node.name}: cubic_coeff_a other than -0.75 is not supported")
            torch_mode = {"linear": ("linear", "bilinear", "trilinear"), "cubic": (None, "bicubic", None)}[mode][rank - 1]
            if torch_mode is None:
                raise ValueError(f"Resize {node.name}: {mode} on {rank}-D data is not supported")
        else:
            raise ValueError(f"Resize {node.name}: mode {mode} is not supported")
        if sizes is not None:
            if list(sizes[:2]) != list(x.shape[:2]):
                raise ValueError(f"Resize {node.name}: resizing the batch or channel dim is not supported")
            return F.interpolate(x, size=sizes[2:], mode=torch_mode, align_corners=align, antialias=antialias)
        if any(s != 1 for s in scales[:2]):
            raise ValueError(f"Resize {node.name}: scaling the batch or channel dim is not supported")
        return F.interpolate(x, scale_factor=scales[2:], mode=torch_mode, align_corners=align,
                             recompute_scale_factor=False, antialias=antialias)

    def op_Resize(self, node, x, roi=None, scales=None, sizes=None):
        if self.opset < 11:
            # opset 10: inputs (X, scales), nearest with asymmetric coordinates and floor
            scales = [float(s) for s in self._value(node, 1, roi)]
            return self._interpolate(node, x, "nearest", "asymmetric", "floor", scales, None, False)
        scales = None if scales is None or scales.numel() == 0 else [float(s) for s in self._value(node, 2, scales)]
        sizes = None if sizes is None or sizes.numel() == 0 else self._ints(node, 3, sizes)
        return self._interpolate(node, x, node.attrs.get("mode", "nearest"),
                                 node.attrs.get("coordinate_transformation_mode", "half_pixel"),
                                 node.attrs.get("nearest_mode", "round_prefer_floor"),
                                 scales, sizes, bool(node.attrs.get("antialias", 0)))

    def op_Upsample(self, node, x, scales=None):
        scales = node.attrs["scales"] if scales is None else [float(s) for s in self._value(node, 1, scales)]
        mode = node.attrs.get("mode", "nearest")
        if mode != "nearest":
            raise ValueError(f"Upsample {node.name}: only nearest mode is supported")
        return self._interpolate(node, x, "nearest", "asymmetric", "floor", scales, None, False)
