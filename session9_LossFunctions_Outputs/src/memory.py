"""Peak tensor memory, measured rather than estimated.

Deliverable 7 asks for the peak memory of two implementations of one loss.
That is harder to measure honestly than it looks, and the easy routes are all
slightly wrong:

* ``torch.cuda.max_memory_allocated`` is exactly right and does not exist on
  this machine (Apple silicon, MPS).
* ``torch.mps.current_allocated_memory`` is a level, not a peak, so reading it
  before and after misses the maximum in between.
* process RSS never goes back down, and includes the interpreter, the weights
  and the corpus.

So this module counts the tensors themselves.  ``TrackTensors`` is a
``TorchDispatchMode``: every operator that runs inside the block reports its
outputs, each newly allocated storage is added to a running total, and a
weakref finalizer subtracts it again the moment the tensor is freed.  The
maximum that running total ever reaches is the peak.

That definition is device-independent, ignores allocator caching, and counts
exactly the thing the session cares about -- how much has to be resident at
once -- including tensors autograd is holding on to for the backward pass,
because those are alive and their finalizers have not fired.

``peak_rss_subprocess`` is the cross-check: a completely separate process, a
completely different mechanism, no shared code with the tracker.  Two methods
that agree are worth more than one method that is clever.
"""

from __future__ import annotations

import subprocess
import sys
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

MIB = 1024 ** 2


class TrackTensors(TorchDispatchMode):
    """Track live tensor bytes allocated by operators inside this block."""

    def __init__(self, baseline: int = 0):
        super().__init__()
        self.live = 0
        self.peak = 0
        self.total_allocated = 0
        self.n_allocations = 0
        self.largest_single = 0
        self.largest_name = ""
        self._live_ptrs: dict[int, int] = {}
        self.baseline = baseline

    def _release(self, ptr, nbytes):
        # A storage's address can be reused after it is freed, so only forget
        # the pointer if it is still recorded with the size we charged for.
        if self._live_ptrs.get(ptr) == nbytes:
            del self._live_ptrs[ptr]
            self.live -= nbytes

    def _charge(self, t, name):
        if not isinstance(t, torch.Tensor) or t.numel() == 0:
            return
        try:
            storage = t.untyped_storage()
            ptr, nbytes = storage.data_ptr(), storage.nbytes()
        except Exception:
            return
        if ptr == 0 or ptr in self._live_ptrs:
            return          # a view of, or an alias for, something already counted
        self._live_ptrs[ptr] = nbytes
        self.live += nbytes
        self.n_allocations += 1
        self.total_allocated += nbytes
        if nbytes > self.largest_single:
            self.largest_single, self.largest_name = nbytes, name
        if self.live > self.peak:
            self.peak = self.live
        weakref.finalize(t, self._release, ptr, nbytes)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = str(func)
        for t in tree_flatten(out)[0]:
            self._charge(t, name)
        return out

    # -- reporting ----------------------------------------------------------
    @property
    def peak_mib(self) -> float:
        return self.peak / MIB

    @property
    def largest_mib(self) -> float:
        return self.largest_single / MIB

    def summary(self) -> dict:
        return {
            "peak_bytes": self.peak,
            "peak_mib": round(self.peak_mib, 3),
            "largest_single_bytes": self.largest_single,
            "largest_single_mib": round(self.largest_mib, 3),
            "largest_single_op": self.largest_name,
            "n_allocations": self.n_allocations,
            "total_allocated_mib": round(self.total_allocated / MIB, 3),
        }


def logits_bytes(n_positions: int, vocab: int, dtype=torch.float32) -> int:
    """What a materialised logits tensor costs, from the shape alone."""
    return n_positions * vocab * torch.tensor([], dtype=dtype).element_size()


PROBE = r"""
import resource, sys, torch
sys.path.insert(0, {root!r})
KIB = 1024 if sys.platform.startswith("linux") else 1        # ru_maxrss units
def rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024) * KIB
{setup}
torch.mps.synchronize() if torch.backends.mps.is_available() else None
before = rss_mib()
{body}
torch.mps.synchronize() if torch.backends.mps.is_available() else None
after = rss_mib()
print("RSS_PEAK_DELTA_MIB", after - before)
"""


def peak_rss_subprocess(root: str, setup: str, body: str) -> float:
    """Run ``setup`` then ``body`` in a fresh process; return peak RSS growth.

    ``ru_maxrss`` is a high-water mark that never falls, so the growth between
    the two readings is the peak the body reached, not its final level.  Coarse
    -- it counts allocator retention and page faults too -- but it is measured
    by the operating system and cannot be talked into agreeing with us.
    """
    src = PROBE.format(root=root, setup=setup, body=body)
    out = subprocess.run([sys.executable, "-c", src], capture_output=True,
                         text=True, timeout=900)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-2000:])
    for line in out.stdout.splitlines():
        if line.startswith("RSS_PEAK_DELTA_MIB"):
            return float(line.split()[1])
    raise RuntimeError(f"no reading in output:\n{out.stdout}\n{out.stderr}")
