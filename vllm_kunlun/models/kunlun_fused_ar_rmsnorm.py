"""
Wire the fused AllReduce+Residual+RMSNorm operator into the Qwen3 inference
forward path (Approach 1: eager rewrite of the decoder layer).

Per-layer, vLLM does:
    hidden = self_attn(...)                       # o_proj is RowParallelLinear -> all_reduce
    hidden, residual = post_attention_layernorm(hidden, residual)  # residual add + RMSNorm

which is exactly AllReduce -> residual add -> RMSNorm. We:
  1. set o_proj.reduce_results=False so the layer receives the *un-reduced*
     per-rank partial, and
  2. replace the post_attention_layernorm step with the fused PG backend op
     `all_reduce_rms_norm(ar_in, residual_in, -> residual_out, norm_out, ...)`.

Gating (from the BKCL microbenchmark + kernel constraints):
  - only TP==4 (the KL3 single-node mesh path),
  - token_num % TP == 0 (mesh reduce_scatter divisibility),
  - 0 < token_num <= KUNLUN_FUSE_AR_RMSNORM_MAX_TOKENS (fused wins for small
    token counts; larger falls back to plain all_reduce + RMSNorm).
Everything else falls back to the exact original behaviour, so correctness is
preserved regardless of gating.

Enable with env KUNLUN_FUSE_AR_RMSNORM=1 (default off).
"""
import os

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_TOKENS = int(os.getenv("KUNLUN_FUSE_AR_RMSNORM_MAX_TOKENS", "512"))
_applied = False
_fused_hits = 0


def _enabled() -> bool:
    return os.getenv("KUNLUN_FUSE_AR_RMSNORM", "0") in ("1", "true", "ON", "on")


def _tp_backend(device: torch.device):
    # ProcessGroupXCCL (kccl backend) exposes the pybind all_reduce_rms_norm.
    return get_tp_group().device_group._get_backend(device)


def apply() -> None:
    global _applied
    if _applied or not _enabled():
        return
    try:
        import vllm.model_executor.models.qwen3 as q3
    except Exception as e:  # pragma: no cover
        logger.warning("[KunlunFuse] qwen3 import failed, skip fusion: %s", e)
        return

    Layer = q3.Qwen3DecoderLayer
    _orig_init = Layer.__init__
    _orig_forward = Layer.forward

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._kunlun_fuse = get_tensor_model_parallel_world_size() == 4
        if self._kunlun_fuse:
            # take the un-reduced partial; the reduce is folded into the norm.
            self.self_attn.o_proj.reduce_results = False

    def _fused_post_attn_norm(self, hidden_partial, residual):
        ln = self.post_attention_layernorm
        m = hidden_partial.shape[0]
        use_fused = (
            getattr(self, "_kunlun_fuse", False)
            and residual is not None
            and 0 < m <= _MAX_TOKENS
            and m % 4 == 0
        )
        if not use_fused:
            # correctness-preserving fallback: we still owe the all_reduce
            # because o_proj skipped it.
            hidden = tensor_model_parallel_all_reduce(hidden_partial)
            return ln(hidden, residual)

        hidden_partial = hidden_partial.contiguous()
        residual = residual.contiguous()
        residual_out = torch.empty_like(residual)
        norm_out = torch.empty_like(hidden_partial)
        work = _tp_backend(hidden_partial.device).all_reduce_rms_norm(
            hidden_partial,
            residual,
            residual_out,
            norm_out,
            ln.weight.data,
            float(ln.variance_epsilon),
            False,  # is_gemma
        )
        # The collective runs on the independent xccl stream (see
        # ProcessGroupXCCL::useXcclStream); order the compute stream after it by
        # waiting on the returned Work, otherwise mlp() races the norm output.
        if work is not None:
            work.wait()
        global _fused_hits
        if _fused_hits < 3:
            logger.info(
                "[KunlunFuse] fused all_reduce_rms_norm HIT tokens=%d hidden=%d",
                m,
                hidden_partial.shape[-1],
            )
        _fused_hits += 1
        return norm_out, residual_out

    def _patched_forward(self, positions, hidden_states, residual):
        if not getattr(self, "_kunlun_fuse", False):
            return _orig_forward(self, positions, hidden_states, residual)
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        # Fused: AllReduce(o_proj partial) + residual + RMSNorm
        hidden_states, residual = _fused_post_attn_norm(self, hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    Layer.__init__ = _patched_init
    Layer.forward = _patched_forward
    _applied = True
    logger.info(
        "[KunlunFuse] fused AllReduce+Residual+RMSNorm wired into Qwen3 "
        "(post-attention site, TP4, max_tokens=%d)",
        _MAX_TOKENS,
    )
