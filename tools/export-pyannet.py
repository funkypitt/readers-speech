#!/usr/bin/env python
"""Writes pyannote's segmentation-3.0 network as plain float32 weights for pyannet.c.

    export-pyannet.py model.onnx src/main/assets/pyannote-segmentation-3.0.bin

The ONNX file is sherpa-onnx's export of `pyannote/segmentation-3.0` (MIT, CNRS 2022); the
sinc filters of the front end are baked in it as a constant. Needs the `onnx` package
(`~/miniconda3/envs/interview/bin/python`).

File layout, all little-endian — the reader is `pyannet_load` in src/main/cpp/pyannet.c:

    char[4]   "PYAN"
    int32     version = 1
    int32[13] sinc_channels (80), sinc_kernel (251), sinc_stride (10),
              conv1_out (60), conv1_kernel (5), conv2_out (60), conv2_kernel (5), pool (3),
              lstm_layers (4), lstm_hidden (128), linear0_out (128), linear1_out (128), classes (7)
    float32   instance-norm epsilon (1e-5), leaky-relu slope (0.01)
    int32     number of float32 weights that follow
    float32[] the weights, in this order:
        wav norm   scale[1], bias[1]                 InstanceNormalization of the raw waveform
        sinc conv  W[80][251]                        no bias; followed by abs, maxpool 3
        norm0      scale[80], bias[80]               then leaky relu
        conv1      W[60][80][5], b[60]               then maxpool 3
        norm1      scale[60], bias[60]               then leaky relu
        conv2      W[60][60][5], b[60]               then maxpool 3
        norm2      scale[60], bias[60]               then leaky relu
        for layer 0..3, for direction forward, backward:
            WT[in][512]   input weights, TRANSPOSED to [input][4*hidden]; in = 60 (layer 0) or 256
            RT[128][512]  recurrent weights, transposed to [hidden][4*hidden]
            b[512]        W-bias + R-bias, summed
          gates in ONNX order along the 512: input, output, forget, cell
        linear0    W[256][128] ([in][out], as ONNX MatMul), b[128]   then leaky relu
        linear1    W[128][128], b[128]                               then leaky relu
        classifier W[128][7], b[7]                                   then log-softmax
"""
import struct
import sys

import numpy as np
import onnx
from onnx import numpy_helper

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
init = {t.name: numpy_helper.to_array(t) for t in g.initializer}
nodes = list(g.node)


def attr(node, name):
    for a in node.attribute:
        if a.name == name:
            return onnx.helper.get_attribute_value(a)
    raise KeyError(name)


def by_type(op):
    return [n for n in nodes if n.op_type == op]


# ── read the graph rather than assume it ─────────────────────────────────────────────────
norms = by_type("InstanceNormalization")
assert len(norms) == 4, norms
eps = {attr(n, "epsilon") for n in norms}
assert len(eps) == 1, eps
eps = eps.pop()
alphas = {attr(n, "alpha") for n in by_type("LeakyRelu")}
assert len(alphas) == 1 and len(by_type("LeakyRelu")) == 5, alphas
alpha = alphas.pop()
pools = by_type("MaxPool")
assert len(pools) == 3 and all(attr(p, "kernel_shape") == [3] and attr(p, "strides") == [3] and attr(p, "pads") == [0, 0] for p in pools)
assert len(by_type("Abs")) == 1
# the sinc conv sits in both branches of an If on the channel count; both are the same conv
sinc = None
for n in by_type("If"):
    for a in n.attribute:
        sub = onnx.helper.get_attribute_value(a)
        (conv,) = [x for x in sub.node if x.op_type == "Conv"]
        assert attr(conv, "strides") == [10] and attr(conv, "kernel_shape") == [251] and attr(conv, "pads") == [0, 0] and len(conv.input) == 2
        w = init[conv.input[1]]
        assert sinc is None or np.array_equal(sinc, w)
        sinc = w
assert sinc.shape == (80, 1, 251)
convs = by_type("Conv")
assert len(convs) == 2 and all(attr(c, "strides") == [1] and attr(c, "kernel_shape") == [5] and attr(c, "pads") == [0, 0] and attr(c, "group") == 1 for c in convs)
conv1_w, conv1_b = init[convs[0].input[1]], init[convs[0].input[2]]
conv2_w, conv2_b = init[convs[1].input[1]], init[convs[1].input[2]]
assert conv1_w.shape == (60, 80, 5) and conv2_w.shape == (60, 60, 5)
lstms = by_type("LSTM")
assert len(lstms) == 4
for l in lstms:
    assert attr(l, "direction") == b"bidirectional" and attr(l, "hidden_size") == 128 and attr(l, "input_forget") == 0
    names = {a.name for a in l.attribute}
    assert not names & {"activations", "activation_alpha", "activation_beta", "clip", "layout"}, names   # defaults: sigmoid, tanh, tanh; layout 0
    assert l.input[4] == ""                                                                              # no sequence lengths
matmuls = by_type("MatMul")
assert len(matmuls) == 3
adds = by_type("Add")
assert len(adds) == 3
(lsm,) = by_type("LogSoftmax")
assert attr(lsm, "axis") == -1
wav_norm = norms[0]
assert wav_norm.input[0] == g.input[0].name
assert sorted(n.op_type for n in nodes if n.op_type not in {"Shape", "Gather", "Equal", "If", "Unsqueeze", "Concat", "ConstantOfShape", "Slice", "Transpose", "Reshape"}) == sorted(
    ["InstanceNormalization"] * 4 + ["Abs"] + ["MaxPool"] * 3 + ["LeakyRelu"] * 5 + ["Conv"] * 2 + ["LSTM"] * 4 + ["MatMul"] * 3 + ["Add"] * 3 + ["LogSoftmax"])

# ── lay the weights out ──────────────────────────────────────────────────────────────────
out = []


def put(a, shape):
    a = np.ascontiguousarray(a, dtype=np.float32)
    assert a.shape == tuple(shape), (a.shape, shape)
    out.append(a.reshape(-1))


put(init[wav_norm.input[1]], (1,)); put(init[wav_norm.input[2]], (1,))
put(sinc[:, 0, :], (80, 251))
put(init[norms[1].input[1]], (80,)); put(init[norms[1].input[2]], (80,))
put(conv1_w, (60, 80, 5)); put(conv1_b, (60,))
put(init[norms[2].input[1]], (60,)); put(init[norms[2].input[2]], (60,))
put(conv2_w, (60, 60, 5)); put(conv2_b, (60,))
put(init[norms[3].input[1]], (60,)); put(init[norms[3].input[2]], (60,))
for li, l in enumerate(lstms):
    W, R, B = init[l.input[1]], init[l.input[2]], init[l.input[3]]
    n_in = 60 if li == 0 else 256
    assert W.shape == (2, 512, n_in) and R.shape == (2, 512, 128) and B.shape == (2, 1024)
    for d in range(2):
        put(W[d].T, (n_in, 512))
        put(R[d].T, (128, 512))
        put(B[d, :512] + B[d, 512:], (512,))
for mm, add, shape in zip(matmuls, adds, [(256, 128), (128, 128), (128, 7)]):
    put(init[mm.input[1]], shape)
    (bias,) = [i for i in add.input if i in init]
    put(init[bias], (shape[1],))

weights = np.concatenate(out)
header = struct.pack("<4si13iffi", b"PYAN", 1, 80, 251, 10, 60, 5, 60, 5, 3, 4, 128, 128, 128, 7, eps, alpha, len(weights))
with open(dst, "wb") as f:
    f.write(header)
    f.write(weights.astype("<f4").tobytes())
print("%s: %d weights, %.2f MB, eps %g, leaky %g" % (dst, len(weights), (len(header) + 4 * len(weights)) / 1e6, eps, alpha))
