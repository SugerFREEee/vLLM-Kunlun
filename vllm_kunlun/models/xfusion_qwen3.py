"""Wire XFusion's fused AllReduce+Residual+RMSNorm into vLLM's Qwen3 decoder.

Model adaptation lives HERE (in vLLM-Kunlun), not in the XFusion library —
XFusion only exposes the op `xfusion.ar_residual_rmsnorm(...)` and decides
fuse-vs-fallback internally. Enable with env XFUSION_ENABLE=1.

Per layer, both norm sites (input_layernorm / post_attention_layernorm) are
routed through XFusion. o_proj/down_proj get reduce_results=False so the
all-reduce is folded into the fused op (the last layer's down_proj keeps its
reduce so the model's final norm still sees a reduced hidden).
"""
import os
import re

from vllm.distributed import (get_tensor_model_parallel_world_size,
                              get_tp_group)
from vllm.logger import init_logger

logger = init_logger(__name__)
_applied = False


def _enabled() -> bool:
    return os.getenv("XFUSION_ENABLE", "0") in ("1", "true", "ON", "on")


def patch_qwen3() -> None:
    global _applied
    if _applied or not _enabled():
        return
    try:
        import xfusion
        import vllm.model_executor.models.qwen3 as q3
    except Exception as e:  # pragma: no cover
        logger.warning("[XFusion] patch skipped: %s", e)
        return

    Layer = q3.Qwen3DecoderLayer
    _orig_init = Layer.__init__
    _orig_forward = Layer.forward

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        config = kwargs.get("config", args[0] if args else None)
        prefix = kwargs.get("prefix", args[3] if len(args) > 3 else "")
        self._xfusion = get_tensor_model_parallel_world_size() == 4
        if not self._xfusion:
            return
        mm = re.search(r"layers\.(\d+)", prefix or "")
        idx = int(mm.group(1)) if mm else -1
        nlayers = getattr(config, "num_hidden_layers", -1)
        self.self_attn.o_proj.reduce_results = False
        if idx != nlayers - 1:
            self.mlp.down_proj.reduce_results = False

    def _norm(ln, hidden, residual):
        import xfusion
        return xfusion.ar_residual_rmsnorm(
            hidden, residual, ln.weight, float(ln.variance_epsilon),
            group=get_tp_group().device_group)

    def _patched_forward(self, positions, hidden_states, residual):
        if not getattr(self, "_xfusion", False):
            return _orig_forward(self, positions, hidden_states, residual)
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = _norm(self.input_layernorm, hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states, residual = _norm(self.post_attention_layernorm, hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    Layer.__init__ = _patched_init
    Layer.forward = _patched_forward
    _applied = True
    logger.warning("[XFusion] wired into Qwen3 (TP4, both norm sites); backend=%s",
                   xfusion.backend())
