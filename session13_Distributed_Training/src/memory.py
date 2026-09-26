"""A peak-memory meter that works on MPS, where there is no max_memory_allocated().

On MPS the meter samples torch.mps.current_allocated_memory() (bytes held by live tensors) and
torch.mps.driver_allocated_memory() (what Metal has handed the process, caching allocator
included) at every probe() and keeps the maximum. The training step probes after the forward
pass, once per layer during the backward pass (a tensor hook for the standard stack, inside the
reversible backward for the others), after backward and after the optimizer step. A sampled peak
can miss a transient inside a single kernel, so it is a lower bound on the true peak; on CUDA the
exact torch.cuda.max_memory_allocated() is used instead.
"""
import torch

_state = {"on": False, "dev": "cpu", "alloc": 0, "driver": 0}


def start(device):
    dev = torch.device(device).type
    _state.update(on=True, dev=dev, alloc=0, driver=0)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    probe()


def active():
    return _state["on"]


def probe():
    if not _state["on"]:
        return
    dev = _state["dev"]
    if dev == "mps":
        _state["alloc"] = max(_state["alloc"], torch.mps.current_allocated_memory())
        _state["driver"] = max(_state["driver"], torch.mps.driver_allocated_memory())
    elif dev == "cuda":
        _state["alloc"] = max(_state["alloc"], torch.cuda.max_memory_allocated())
        _state["driver"] = max(_state["driver"], torch.cuda.max_memory_reserved())


def stop():
    probe()
    _state["on"] = False
    return {"peak_alloc_bytes": int(_state["alloc"]), "peak_driver_bytes": int(_state["driver"])}


def now():
    dev = _state["dev"]
    if dev == "mps":
        return torch.mps.current_allocated_memory()
    if dev == "cuda":
        return torch.cuda.memory_allocated()
    return 0


class OpLevelPeak:
    """Exact peak of live tensor bytes, sampled after every aten op (forward and backward).

    A TorchDispatchMode sees every operator the step runs, including the ones autograd runs in the
    backward pass, so the maximum of current_allocated_memory() over those sample points is the
    true peak up to a single kernel's internal workspace. It costs a Python call per op, so it is
    only switched on for a short audit, never while throughput is being timed.
    """

    def __init__(self, device, sync=False):
        from torch.utils._python_dispatch import TorchDispatchMode

        dev = torch.device(device).type
        read = {"mps": torch.mps.current_allocated_memory,
                "cuda": torch.cuda.memory_allocated}.get(dev, lambda: 0)
        if sync and dev == "mps":
            # MPS keeps a freed buffer counted until the command buffer using it completes, so an
            # unsynchronised reading depends on how far the GPU lags the CPU. Syncing after every op
            # makes the reading the live-tensor total at that op, and repeatable.
            def read():
                torch.mps.synchronize()
                return torch.mps.current_allocated_memory()
        meter = self
        self.peak, self.ops, self.where = 0, 0, None

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                cur = read()
                meter.ops += 1
                if cur > meter.peak:
                    meter.peak, meter.where = cur, str(func)
                return out

        self.mode = Mode()

    def __enter__(self):
        self.mode.__enter__()
        return self

    def __exit__(self, *exc):
        return self.mode.__exit__(*exc)
