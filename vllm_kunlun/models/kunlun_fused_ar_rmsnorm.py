"""
Wire the fused AllReduce+Residual+RMSNorm operator into the Qwen3 inference
forward path (Approach 1: eager rewrite of the decoder layer).

Per layer, vLLM does TWO AllReduce -> residual add -> RMSNorm sites:
  * input_layernorm(hidden, residual)          <- previous layer's down_proj all_reduce
  * post_attention_layernorm(hidden, residual)  <- this layer's o_proj all_reduce
Both are exactly what backend.all_reduce_rms_norm fuses.

We set o_proj.reduce_results=False (all layers) and down_proj.reduce_results=False
(all layers EXCEPT the last one, so the model's final RMSNorm still gets a fully
reduced hidden and stays a plain norm -- no need to patch Qwen2Model.forward).
Then each `layernorm(hidden, residual)` is routed through the fused op:
  * post_attention_layernorm: always (o_proj partial)
  * input_layernorm: whenever residual is not None (previous down_proj partial);
    the very first layer has residual=None (embedding, already full) -> plain norm.

Gating (BKCL microbenchmark + kernel constraints): TP==4,
0<token_num<=KUNLUN_FUSE_AR_RMSNORM_MAX_TOKENS. token_num need NOT be divisible
by the TP size: the BKCL mesh rs_norm_2ag path pads it up to a multiple of nranks
internally. Otherwise fall back to a plain all_reduce (we still owe it, since the
linear skipped it) + RMSNorm, so results are correct regardless of gating.

Enable with env KUNLUN_FUSE_AR_RMSNORM=1 (default off).
"""
import os
import re
import time

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_MIN_TOKENS = int(os.getenv("KUNLUN_FUSE_AR_RMSNORM_MIN_TOKENS", "8"))
_MAX_TOKENS = int(os.getenv("KUNLUN_FUSE_AR_RMSNORM_MAX_TOKENS", "64"))
_applied = False
_fused_hits = 0
_backend_cache = {}


def _enabled() -> bool:
    return os.getenv("KUNLUN_FUSE_AR_RMSNORM", "0") in ("1", "true", "ON", "on")


def _tp_backend(device: torch.device):
    # Cache the ProcessGroupXCCL backend per device: _get_backend does a lookup
    # that is far too expensive to repeat 2x per layer per decode step.
    b = _backend_cache.get(device.index)
    if b is None:
        b = get_tp_group().device_group._get_backend(device)
        _backend_cache[device.index] = b
    return b


_norm_buf_cache = {}


def _get_norm_buf(x: torch.Tensor) -> torch.Tensor:
    # Reused per-shape output buffer for norm_out (avoids per-call empty_like).
    # Safe: norm_out is consumed by the following op (attn/mlp) on the same
    # stream before the next _fused_norm reuses this buffer.
    key = (x.shape[0], x.shape[1], x.dtype, x.device.index)
    b = _norm_buf_cache.get(key)
    if b is None:
        b = torch.empty_like(x)
        _norm_buf_cache[key] = b
    return b


_residual_pool = {}


def _next_residual_buf(x: torch.Tensor) -> torch.Tensor:
    # Ping-pong pool of 2 buffers per shape for residual_out. The residual is a
    # threaded accumulator: call k reads residual_in (== call k-1's residual_out)
    # and must write a DISTINCT buffer (the mesh op forbids residual_out aliasing
    # residual_in). Alternating between 2 persistent buffers guarantees
    # residual_out != residual_in with zero per-call allocation; each buffer's
    # value is consumed one step before it is overwritten two steps later.
    key = (x.shape[0], x.shape[1], x.dtype, x.device.index)
    p = _residual_pool.get(key)
    if p is None:
        p = [[torch.empty_like(x), torch.empty_like(x)], 0]
        _residual_pool[key] = p
    b = p[0][p[1]]
    p[1] ^= 1
    return b


import atexit  # noqa: E402

_TIMING = os.getenv("KUNLUN_FUSE_TIMING", "0") in ("1", "true", "on")
_seg_ms = 0.0
_seg_n = 0


def _report_timing():
    if _seg_n:
        logger.info(
            "[KunlunFuse][TIMING] mode=%s seg_calls=%d total_seg=%.1fms avg=%.4fms",
            os.getenv("KUNLUN_FUSE_MODE", "fused"), _seg_n, _seg_ms, _seg_ms / _seg_n)


atexit.register(_report_timing)


def _run_timed(fn):
    """Optionally measure the on-device time of the AllReduce+Residual+RMSNorm
    segment (env KUNLUN_FUSE_TIMING=1). Per-call sync perturbs end-to-end tok/s
    but yields the accumulated segment device-time for attribution."""
    global _seg_ms, _seg_n
    if not _TIMING:
        return fn()
    # CUDA-event elapsed_time returns 0 on XPU; use host wall-clock with syncs.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    _seg_ms += (time.perf_counter() - t0) * 1000.0
    _seg_n += 1
    if _seg_n % 2000 == 0:
        logger.info("[KunlunFuse][TIMING] mode=%s seg_calls=%d total_seg=%.1fms avg=%.4fms",
                    os.getenv("KUNLUN_FUSE_MODE", "fused"), _seg_n, _seg_ms, _seg_ms / _seg_n)
    return r


def _fused_norm(self, ln, hidden_partial, residual):
    """residual_out = AllReduce(hidden_partial) + residual; norm_out = RMSNorm(residual_out).

    Gated to the empirically-measured winning range M in [MIN, MAX] (crossover
    bench, hidden=5120 TP4): fused wins ~1.3-2.7x for M~8..128 and loses for
    M<=4 and M>=256, so fall back to plain all_reduce + RMSNorm outside it.
    """
    global _fused_hits
    m = hidden_partial.shape[0]
    use_fused = (
        getattr(self, "_kunlun_fuse", False)
        and residual is not None
        and _MIN_TOKENS <= m <= _MAX_TOKENS
    )
    if not use_fused:
        # linear skipped the reduce (reduce_results=False), so we must do it here.
        hidden = tensor_model_parallel_all_reduce(hidden_partial)
        return ln(hidden, residual)

    if not hidden_partial.is_contiguous():
        hidden_partial = hidden_partial.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()

    _mode = os.getenv("KUNLUN_FUSE_MODE", "fused")

    # Ablation: baseline ops, fully IN-PLACE (no empty_like), like BASE does.
    # vs BASE  => cost of reduce_results=False + patched-forward structure.
    # vs baseline_ops => cost of the 2x empty_like allocations.
    if _mode == "baseline_ops_inplace":
        def _op():
            reduced = tensor_model_parallel_all_reduce(hidden_partial)
            torch.ops._C.add_rmsnorm(
                reduced, residual, residual_output=residual,
                weight=ln.weight.data, eps=ln.variance_epsilon, output=reduced)
            return reduced
        reduced = _run_timed(_op)
        return reduced, residual

    # Attribution mode: same integration but BASELINE ops with empty_like outputs.
    if _mode == "baseline_ops":
        residual_out = torch.empty_like(residual)
        norm_out = torch.empty_like(hidden_partial)
        reduced = tensor_model_parallel_all_reduce(hidden_partial)
        torch.ops._C.add_rmsnorm(
            reduced, residual, residual_output=residual_out,
            weight=ln.weight.data, eps=ln.variance_epsilon, output=norm_out)
        return norm_out, residual_out

    # DEFAULT fused path. Zero per-call allocation, no aliasing:
    #   norm_out     -> reused per-shape buffer (consumed immediately)
    #   residual_out -> ping-pong pool buffer (!= residual_in, threaded stream)
    # The mesh op requires both outputs distinct from ar_in and residual_in.
    norm_out = _get_norm_buf(hidden_partial)
    residual_out = _next_residual_buf(residual)

    def _op():
        w = _tp_backend(hidden_partial.device).all_reduce_rms_norm(
            hidden_partial, residual, residual_out, norm_out,
            ln.weight.data, float(ln.variance_epsilon), False,
        )
        if w is not None:
            w.wait()
    _run_timed(_op)
    if _fused_hits < 2:
        logger.info("[KunlunFuse] fused HIT tokens=%d hidden=%d", m, hidden_partial.shape[-1])
    _fused_hits += 1
    return norm_out, residual_out


def apply() -> None:
    global _applied
    if _applied or not _enabled():
        return
    try:
        import vllm.model_executor.models.qwen3 as q3
    except Exception as e:  # pragma: no cover
        logger.warning("[KunlunFuse] qwen3 import failed, skip: %s", e)
        return

    Layer = q3.Qwen3DecoderLayer
    _orig_init = Layer.__init__
    _orig_forward = Layer.forward

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        config = kwargs.get("config", args[0] if args else None)
        prefix = kwargs.get("prefix", args[3] if len(args) > 3 else "")
        self._kunlun_fuse = get_tensor_model_parallel_world_size() == 4
        if not self._kunlun_fuse:
            return
        mm = re.search(r"layers\.(\d+)", prefix or "")
        idx = int(mm.group(1)) if mm else -1
        nlayers = getattr(config, "num_hidden_layers", -1)
        is_last = (idx == nlayers - 1)
        # o_proj -> post_attention_layernorm fusion (every layer)
        self.self_attn.o_proj.reduce_results = False
        # down_proj -> next layer's input_layernorm fusion; keep the LAST layer
        # reduced so the model's final norm stays a plain RMSNorm.
        if not is_last:
            self.mlp.down_proj.reduce_results = False

    def _patched_forward(self, positions, hidden_states, residual):
        if not getattr(self, "_kunlun_fuse", False):
            return _orig_forward(self, positions, hidden_states, residual)
        # input_layernorm (residual!=None => previous down_proj partial)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = _fused_norm(self, self.input_layernorm, hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        # post_attention_layernorm (o_proj partial)
        hidden_states, residual = _fused_norm(self, self.post_attention_layernorm, hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    Layer.__init__ = _patched_init
    Layer.forward = _patched_forward
    _applied = True
    logger.info(
        "[KunlunFuse] fused AR+Residual+RMSNorm wired into Qwen3 (both sites, "
        "TP4, fuse when %d<=tokens<=%d)",
        _MIN_TOKENS, _MAX_TOKENS,
    )
