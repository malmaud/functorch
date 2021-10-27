import torch
from torch import fx
from torch import nn as nn
from torch.nn import functional as F
from functorch import compiled_function, partition_with_recompute_fwd_in_bwd
import statistics
import inspect
from typing import Callable
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.fx import replace_pattern


database = {}
database['cpu'] = {}
database['cuda'] = {}

def _num_args(fn: Callable):
    return len(inspect.signature(fn).parameters)

def gelu_bias(bias, y):
    x = bias + y
    return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

def swish(x):
    return x * torch.sigmoid(x)

def mish(x):
    return x.mul(torch.tanh(F.softplus(x)))

def hard_sigmoid(x):
    return (x + 3).clamp(min=0, max=6).div(6.)

def hard_swish(x):
    return x * (x + 3).clamp(min=0, max=6).div(6.)

def hard_mish(x):
    return 0.5 * x * (x + 2).clamp(min=0, max=2)

def timeme(fn_name, compiler_name, fn, args, shape, device):
    if device == 'cuda':
        timeme_cuda(fn_name, compiler_name, fn, args, shape)
    else:
        timeme_cpu(fn_name, compiler_name, fn, args, shape)

def timeme_cuda(fn_name, compiler_name, fn, args, shape):
    import time

    warmup = 50
    repeats = 500
    for _ in range(0, warmup):
        ref = fn(*args)
        ref.sum().backward()
    torch.cuda.synchronize()

    fwd_times = []
    bwd_times = [] 
    for _ in range(0, repeats):
        fwd_start = time.time()
        ref = fn(*args)
        torch.cuda.synchronize()
        fwd_end = time.time()

        loss = ref.sum()
        torch.cuda.synchronize()

        bwd_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        bwd_end = time.time()

        fwd_times.append(fwd_end - fwd_start)
        bwd_times.append(bwd_end - bwd_start)
    avg_fwd = round(statistics.mean(fwd_times) * 10**6, 2)
    avg_bwd = round(statistics.mean(bwd_times) * 10**6, 2)
    avg_total = round(avg_fwd + avg_bwd, 2)
    # print(name, "GPU Forward", avg_fwd)
    # print(name, "GPU Backward", avg_bwd)
    # print(name, "GPU Fwd + Bwd", avg_total)
    database['cuda'][fn_name][compiler_name][shape[0]] = (avg_fwd, avg_bwd, avg_total)
    print(fn_name, compiler_name, shape[0], avg_fwd, avg_bwd, avg_total, sep='\t', flush=True)

def timeme_cpu(fn_name, compiler_name, fn, args, shape):
    import time

    warmup = 50
    repeats = 500
    for _ in range(0, warmup):
        ref = fn(*args)
        ref.sum().backward()

    fwd_times = []
    bwd_times = [] 
    for _ in range(0, repeats):
        fwd_start = time.time()
        ref = fn(*args)
        fwd_end = time.time()

        loss = ref.sum()

        bwd_start = time.time()
        loss.backward()
        bwd_end = time.time()

        fwd_times.append(fwd_end - fwd_start)
        bwd_times.append(bwd_end - bwd_start)
    avg_fwd = round(statistics.mean(fwd_times) * 10**6, 2)
    avg_bwd = round(statistics.mean(bwd_times) * 10**6, 2)
    avg_total = round(avg_fwd + avg_bwd, 2)
    # print(name, "Forward", avg_fwd)
    # print(name, "Backward", avg_bwd)
    # print(name, "Fwd + Bwd", avg_total)
    database['cpu'][fn_name][compiler_name][shape[0]] = (avg_fwd, avg_bwd, avg_total)
    print(fn_name, compiler_name, shape[0], avg_fwd, avg_bwd, avg_total, sep='\t', flush=True)

def tanh_backward_decomposition(out_grad, y):
    return torch.sub(out_grad, out_grad * y * y)

def sigmoid_backward_decomposition(out_grad, y):
    # return out_grad * (y * (_create_constant(1.0, torch.float32) - y))
    # return out_grad * (y * (1 - y))
    return out_grad * (y - y * y)

def softplus_backward_decomposition(out_grad, x, beta, threshold, out):
    return out_grad * torch.sigmoid(x)

def noop(x):
    return x

decomposition_rules = {}
decomposition_rules[torch.ops.aten.tanh_backward] = tanh_backward_decomposition
decomposition_rules[torch.ops.aten.sigmoid_backward] = sigmoid_backward_decomposition
decomposition_rules[torch.ops.aten.softplus_backward] = softplus_backward_decomposition
decomposition_rules[torch.ops.aten.detach] = noop

    
def decompose(fx_module):
    """
    Decompose `model` into smaller constituent operations.
    Currently,this only supports decomposing ReLU into its
    mathematical definition: (x > 0) * x
    """
    graph = fx_module.graph
    new_graph = fx.Graph()
    env = {}
    for node in graph.nodes:
        if node.op == 'call_function' and node.target in decomposition_rules:
            # By wrapping the arguments with proxies,
            # we can dispatch to the appropriate
            # decomposition rule and implicitly add it
            # to the Graph by symbolically tracing it.
            proxy_args = [
                fx.Proxy(env[x.name]) if isinstance(x, fx.Node) else x for x in node.args]
            output_proxy = decomposition_rules[node.target](*proxy_args)

            # Operations on `Proxy` always yield new `Proxy`s, and the
            # return value of our decomposition rule is no exception.
            # We need to extract the underlying `Node` from the `Proxy`
            # to use it in subsequent iterations of this transform.
            new_node = output_proxy.node
            env[node.name] = new_node
        else:
            # Default case: we don't have a decomposition rule for this
            # node, so just copy the node over into the new graph.
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
    return fx.GraphModule(fx_module, new_graph)


def ts_operator_authoring(fn):
    def fwd_compile(fx_module, flat_args):
        fx_module = decompose(fx_module)
        traced_module = torch.jit.trace(fx_module, flat_args)
        frozen_module = torch.jit.freeze(traced_module.eval())
        return frozen_module
        # jit_module = torch.jit.trace(fx_module, flat_args)
        # return jit_module
        # print()
        # print(jit_module.graph)
        # torch._C._te.remove_unused_self_argument(jit_module.graph)
        # print(jit_module.graph)
        # te_kernel = torch._C._te.TensorExprKernel(jit_module.graph)
        # print(te_kernel)
        # return fx_module

    def bwd_compile(fx_module, flat_args):
        fx_module = decompose(fx_module)
        traced_module = torch.jit.trace(fx_module, flat_args)
        frozen_module = torch.jit.freeze(traced_module.eval())
        return frozen_module

    compiled_fn = compiled_function(fn, fwd_compile, bwd_compile, partition_with_recompute_fwd_in_bwd)
    return compiled_fn

LOW = 4
HIGH = 28
FUNCTIONS = [gelu_bias, swish, mish, hard_sigmoid, hard_swish, hard_mish]
device = 'cuda'

# HIGH = 5
# FUNCTIONS = [hard_mish]
SHAPES = [(2**x, ) for x in range(LOW, HIGH)]

def parse_database(device):
    for fn_id, fn in enumerate(FUNCTIONS):
        fig, axs = plt.subplots(3, figsize=(10,10), sharex=True)
        fig.suptitle(f"{fn.__name__} on {device}")

        x = range(LOW, HIGH)
        db = database[device][fn.__name__]
        baseline_db = db["EagerBaseline"]
        ts_jit_db = db["Torchscript"]
        op_authoring_db = db["OperatorAuthoring"]
        def _extract_latency(d):
            return [d[x] for x in sorted(d.keys())]
        baseline = _extract_latency(baseline_db)
        ts_jit = _extract_latency(ts_jit_db)
        op_authoring = _extract_latency(op_authoring_db)

        ylabels = ["Fwd latency (ms)", "Bwd latency (ms)", "Fwd + Bwd latency (ms)"]
        for idx in range(3):
            b = [x[idx] for x in baseline]
            t = [x[idx] for x in ts_jit]
            o = [x[idx] for x in op_authoring]
            print(fn.__name__, fn_id, b)
            axs[idx].set_ylim(0, max([max(b), max(t), max(o)]) * 1.1)
            axs[idx].set_xlim(0, HIGH + 1)
            axs[idx].set_ylabel(ylabels[idx])
            lines = axs[idx].plot(x, b, 'b+--', x, t, 'g+--', x, o, 'r+--')

        # handles, labels = axs[2].get_legend_handles_labels()
        labels = ["EagerBaseline", "Torchscript", "OpaeratorAuthoring"]
        fig.legend(lines, labels, loc='upper left')
        plt.xlabel("Size (log2 scale)")
        fig.tight_layout()
        plt.savefig(fn.__name__ + ".png")

def benchmark():
    for fn in FUNCTIONS:
        database[device][fn.__name__] = {}
        database[device][fn.__name__]["EagerBaseline"] = {}
        database[device][fn.__name__]["Torchscript"] = {}
        database[device][fn.__name__]["OperatorAuthoring"] = {}
        for shape in SHAPES:
            if _num_args(fn) == 1:  
                ref_a = torch.rand(*shape, device=device, requires_grad=True)
                args = (ref_a,)
            elif _num_args(fn) == 2:
                ref_a = torch.rand(*shape, device=device, requires_grad=True)
                ref_b = torch.rand(*shape, device=device, requires_grad=True)
                args = (ref_a, ref_b)

            timeme(fn.__name__, "EagerBaseline", fn, args, shape, device)
    
        for shape in SHAPES:
            if _num_args(fn) == 1:  
                ref_a = torch.rand(*shape, device=device, requires_grad=True)
                args = (ref_a,)
            elif _num_args(fn) == 2:
                ref_a = torch.rand(*shape, device=device, requires_grad=True)
                ref_b = torch.rand(*shape, device=device, requires_grad=True)
                args = (ref_a, ref_b)

            traced_fn = torch.jit.trace(fn, args)
            timeme(fn.__name__, "Torchscript", traced_fn, args, shape, device)
 
        for shape in SHAPES:
            if _num_args(fn) == 1:  
                ref_a = torch.rand(*shape, device=device, requires_grad=True)
                args = (ref_a,)
            elif _num_args(fn) == 2:
                ref_a = torch.rand(*shape, device=device, requires_grad=True)
                ref_b = torch.rand(*shape, device=device, requires_grad=True)
                args = (ref_a, ref_b)

            pointwise_fn = ts_operator_authoring(fn)
            name = '\t'.join([fn.__name__, "OperatorAuthoring"])
            timeme(fn.__name__, "OperatorAuthoring", pointwise_fn, args, shape, device)
    parse_database(device)

if __name__ == '__main__':
    benchmark()