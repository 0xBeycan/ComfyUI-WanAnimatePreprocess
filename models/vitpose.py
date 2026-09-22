"""ViTPose as a plain torch module, built from the weights of its ONNX export.

The exported graph spells the transformer out op by op (2257 nodes for ViTPose-H: every
LayerNorm is ReduceMean/Sub/Pow/Sqrt/Div, every GELU is Div/Erf/Add/Mul). Running that
through the generic executor works but pays a python round trip per node; this module
runs the same weights through addmm, F.layer_norm, F.gelu and scaled_dot_product_attention.

`build_vitpose` reads the architecture from the graph: depth and widths from the weight
shapes, the head count from the qkv reshape, the attention scale, LayerNorm epsilon and
the patch-embedding / deconvolution hyper-parameters from the node attributes. A graph
that does not have this structure gets None back and runs through the generic executor.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """x @ W + b with W kept as the ONNX MatMul stores it, [in, out]; no transposed copy on load."""

    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self.weight).reshape(*x.shape[:-1], -1)


class Block(nn.Module):
    def __init__(self, norm1, qkv, proj, norm2, fc1, fc2, heads, scale, gelu):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.gelu = gelu
        self.norm1 = norm1
        self.qkv = qkv
        self.proj = proj
        self.norm2 = norm2
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=self.scale)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, C))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate=self.gelu))


class ViTPoseNet(nn.Module):
    def __init__(self, patch_embed, pos_embed, blocks, last_norm, head):
        super().__init__()
        self.patch_embed = patch_embed
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        self.blocks = nn.ModuleList(blocks)
        self.last_norm = last_norm
        self.head = head

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.last_norm(x).transpose(1, 2).reshape(B, C, H, W)
        return self.head(x)


def _layer_norm(weight, bias, eps):
    layer = nn.LayerNorm(weight.shape[0], eps=eps)
    layer.weight = nn.Parameter(weight, requires_grad=False)
    layer.bias = nn.Parameter(bias, requires_grad=False)
    return layer


def _symmetric(pads):
    half = len(pads) // 2
    return pads[:half] if pads[:half] == pads[half:] else None


def build_vitpose(graph):
    """ViTPoseNet from a ViTPose ONNX export, or None when the graph is not one."""
    t = graph.tensors
    depth = 0
    while f"backbone.blocks.{depth}.norm1.weight" in t:
        depth += 1
    if depth == 0 or "backbone.patch_embed.proj.weight" not in t or "backbone.last_norm.weight" not in t:
        return None
    dim = t["backbone.last_norm.weight"].shape[0]

    # MatMul weights are anonymous; the Add that applies the named bias identifies each Linear
    linears = {}
    for node in graph.nodes:
        if node.op != "MatMul" or node.inputs[1] not in t:
            continue
        for add in graph.consumers(node.outputs[0]):
            bias = next((n for n in add.inputs if n in t and n.endswith(".bias")), None)
            if add.op == "Add" and bias:
                linears[bias[:-len(".bias")]] = (t[node.inputs[1]], t[bias])
    names = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
    if any(f"backbone.blocks.{i}.{n}" not in linears for i in range(depth) for n in names):
        return None

    # patch embedding: the Conv on the patch weight
    conv_node = next((n for n in graph.nodes if n.op == "Conv" and n.inputs[1] == "backbone.patch_embed.proj.weight"), None)
    if conv_node is None:
        return None
    pads = _symmetric(conv_node.attrs.get("pads", [0, 0, 0, 0]))
    if pads is None or conv_node.attrs.get("group", 1) != 1:
        return None
    w = t["backbone.patch_embed.proj.weight"]
    patch_embed = nn.Conv2d(w.shape[1], w.shape[0], w.shape[2:], stride=conv_node.attrs.get("strides", [1, 1]),
                            padding=pads, dilation=conv_node.attrs.get("dilations", [1, 1]))
    patch_embed.weight = nn.Parameter(w, requires_grad=False)
    patch_embed.bias = nn.Parameter(t["backbone.patch_embed.proj.bias"], requires_grad=False)

    # position embedding: the constant [1, N, C] / [1, 1, C] terms added to the tokens
    pos_terms = [t[n] for node in graph.nodes if node.op == "Add"
                 for n in node.inputs if n in t and t[n].dim() == 3 and t[n].shape[-1] == dim]
    if not pos_terms:
        return None
    pos_embed = pos_terms[0]
    for term in pos_terms[1:]:
        pos_embed = pos_embed + term

    # heads from the qkv reshape [B, N, 3, heads, head_dim], scale from the q multiply
    heads, scale = None, None
    qkv_add = next(n for n in graph.nodes if n.op == "Add" and "backbone.blocks.0.attn.qkv.bias" in n.inputs)
    for reshape in graph.consumers(qkv_add.outputs[0]):
        if reshape.op != "Reshape":
            continue
        shape = _shape_values(graph, reshape.inputs[1])
        if 3 in shape and shape.index(3) + 1 < len(shape):
            heads = shape[shape.index(3) + 1]
    for node in graph.nodes:
        if node.op == "Mul" and node.name.startswith("/backbone/blocks.0/attn/"):
            const = next((t[n] for n in node.inputs if n in t and t[n].numel() == 1), None)
            if const is not None:
                scale = float(const)
                break
    if heads is None or dim % heads:
        return None

    eps = 1e-6
    sqrt = next((n for n in graph.nodes if n.op == "Sqrt"), None)
    if sqrt is not None:
        add = graph.producer(sqrt.inputs[0])
        if add is not None and add.op == "Add":
            eps = float(next(t[n] for n in add.inputs if n in t))
    gelu = "none" if any(n.op == "Erf" for n in graph.nodes) else "tanh"

    blocks = []
    for i in range(depth):
        p = f"backbone.blocks.{i}."
        blocks.append(Block(_layer_norm(t[p + "norm1.weight"], t[p + "norm1.bias"], eps),
                            Linear(*linears[p + "attn.qkv"]), Linear(*linears[p + "attn.proj"]),
                            _layer_norm(t[p + "norm2.weight"], t[p + "norm2.bias"], eps),
                            Linear(*linears[p + "mlp.fc1"]), Linear(*linears[p + "mlp.fc2"]),
                            heads, scale, gelu))
    last_norm = _layer_norm(t["backbone.last_norm.weight"], t["backbone.last_norm.bias"], eps)

    head = _build_head(graph, t)
    if head is None:
        return None
    return ViTPoseNet(patch_embed, pos_embed, blocks, last_norm, head)


def _shape_values(graph, name):
    """Constant entries of a shape tensor (None for runtime dims), through a Concat if needed."""
    if name in graph.tensors:
        return [int(v) for v in graph.tensors[name].reshape(-1).tolist()]
    node = graph.producer(name)
    if node is None or node.op != "Concat":
        return []
    values = []
    for inp in node.inputs:
        if inp in graph.tensors:
            values += [int(v) for v in graph.tensors[inp].reshape(-1).tolist()]
        else:
            values.append(None)
    return values


def _build_head(graph, t):
    """The keypoint head: the ConvTranspose / BatchNormalization / Relu / Conv chain after the backbone."""
    start = next((i for i, n in enumerate(graph.nodes) if n.op == "ConvTranspose"), None)
    if start is None:
        return None
    layers = []
    for node in graph.nodes[start:]:
        if node.op == "ConvTranspose":
            pads = _symmetric(node.attrs.get("pads", [0, 0, 0, 0]))
            w = t.get(node.inputs[1])
            if pads is None or w is None or node.attrs.get("group", 1) != 1 or "output_shape" in node.attrs:
                return None
            layer = nn.ConvTranspose2d(w.shape[0], w.shape[1], w.shape[2:], stride=node.attrs.get("strides", [1, 1]),
                                       padding=pads, output_padding=node.attrs.get("output_padding", [0, 0]),
                                       dilation=node.attrs.get("dilations", [1, 1]), bias=len(node.inputs) > 2)
            layer.weight = nn.Parameter(w, requires_grad=False)
            if len(node.inputs) > 2:
                layer.bias = nn.Parameter(t[node.inputs[2]], requires_grad=False)
        elif node.op == "Conv":
            pads = _symmetric(node.attrs.get("pads", [0, 0, 0, 0]))
            w = t.get(node.inputs[1])
            if pads is None or w is None:
                return None
            layer = nn.Conv2d(w.shape[1], w.shape[0], w.shape[2:], stride=node.attrs.get("strides", [1, 1]),
                              padding=pads, dilation=node.attrs.get("dilations", [1, 1]),
                              groups=node.attrs.get("group", 1), bias=len(node.inputs) > 2)
            layer.weight = nn.Parameter(w, requires_grad=False)
            if len(node.inputs) > 2:
                layer.bias = nn.Parameter(t[node.inputs[2]], requires_grad=False)
        elif node.op == "BatchNormalization":
            scale, bias, mean, var = (t.get(n) for n in node.inputs[1:5])
            if any(v is None for v in (scale, bias, mean, var)):
                return None
            layer = nn.BatchNorm2d(scale.shape[0], eps=node.attrs.get("epsilon", 1e-5))
            layer.weight = nn.Parameter(scale, requires_grad=False)
            layer.bias = nn.Parameter(bias, requires_grad=False)
            layer.running_mean.copy_(mean)
            layer.running_var.copy_(var)
        elif node.op == "Relu":
            layer = nn.ReLU()
        else:
            return None
        layers.append(layer)
        if node.outputs[0] in graph.outputs:
            return nn.Sequential(*layers)
    return None

