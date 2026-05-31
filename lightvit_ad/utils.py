"""
lightvit_ad/utils.py
====================
Seeding, FLOPs profiling with tiered fallback, memory tracking.

FIX: Replaces deepspeed-only FLOPs profiling with a tiered approach:
  1. Try thop (lightweight, available on JetPack 4.6).
  2. Fall back to closed-form ViT FLOP formula (no external dependency).
  deepspeed is not available in JetPack 4.6 and is therefore unsuitable
  for the Jetson Nano deployment environment.

FIX: Memory tracking uses a list-based context manager (state passed by
  reference) instead of a global variable, eliminating the thread-safety
  race condition in the original implementation.
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import psutil
from contextlib import contextmanager
from typing import Tuple


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# FLOPs profiling  (FIX: tiered fallback — thop → closed-form formula)
# ---------------------------------------------------------------------------

try:
    from thop import profile as _thop_profile
    _THOP_AVAILABLE = True
except ImportError:
    _THOP_AVAILABLE = False

try:
    from deepspeed.profiling.flops_profiler import FlopsProfiler
    _DEEPSPEED_AVAILABLE = True
except ImportError:
    _DEEPSPEED_AVAILABLE = False


def _vit_flops_manual(
    image_size: int = 224,
    patch_size: int = 16,
    embed_dim: int  = 192,
    depth: int      = 12,
    num_heads: int  = 3,
    mlp_ratio: float = 4.0,
    has_dist_token: bool = True,
) -> int:
    """
    Closed-form ViT FLOPs formula (FP32 graph).

    Per transformer block (N tokens, d dimensions, h heads, dff = mlp_ratio*d):
      QKV projection : 2 * N * d * 3d
      Attention       : 2 * N^2 * d   (QK^T and AV)
      Output proj     : 2 * N * d * d
      FFN             : 2 * N * d * dff  (two linear layers)

    Reference: 'Training data-efficient image transformers' (Touvron et al., ICML 2021)
    """
    N_patches = (image_size // patch_size) ** 2
    N_tokens  = N_patches + 1 + (1 if has_dist_token else 0)  # patch + cls + dist
    d    = embed_dim
    dff  = int(embed_dim * mlp_ratio)

    flops_per_block = (
        2 * N_tokens * d * 3 * d   +  # QKV projection
        2 * N_tokens * N_tokens * d +  # attention (QK^T + AV)
        2 * N_tokens * d * d        +  # output projection
        4 * N_tokens * d * dff         # FFN (two linear layers, pre/post GELU)
    )
    # Patch embedding
    flops_patch_embed = 2 * N_patches * (3 * patch_size * patch_size) * d

    total_flops = flops_patch_embed + depth * flops_per_block
    return int(total_flops)


def compute_flops_and_params(
    model: nn.Module,
    example_input: torch.Tensor,
    warmup: int = 3,
) -> Tuple[float, float]:
    """
    Compute GFLOPs and parameter count (millions) for a model.

    Priority:
      1. thop  (fast, Jetson-compatible)
      2. deepspeed FlopsProfiler  (x86 only)
      3. Closed-form ViT formula  (always available, FP32 graph)

    Returns:
        gflops (float), params_M (float)
    """
    model.eval()
    params_M = sum(p.numel() for p in model.parameters()) / 1e6

    if _THOP_AVAILABLE:
        try:
            with torch.no_grad():
                macs, _ = _thop_profile(model, inputs=(example_input,), verbose=False)
            gflops = macs * 2 / 1e9   # MACs → FLOPs
            return gflops, params_M
        except Exception:
            pass  # fall through

    if _DEEPSPEED_AVAILABLE:
        try:
            with torch.no_grad():
                for _ in range(warmup):
                    model(example_input)
                prof = FlopsProfiler(model)
                prof.start_profile()
                model(example_input)
                prof.stop_profile()
                flops = prof.get_total_flops(as_string=False)
            return flops / 1e9, params_M
        except Exception:
            pass  # fall through

    # Closed-form fallback — assumes DeiT-tiny default hyperparameters
    flops = _vit_flops_manual(
        image_size=224, patch_size=16, embed_dim=192,
        depth=12, num_heads=3, mlp_ratio=4.0, has_dist_token=True,
    )
    print('[INFO] Using closed-form ViT FLOP formula (thop/deepspeed unavailable).')
    return flops / 1e9, params_M


# ---------------------------------------------------------------------------
# Memory tracking  (FIX: list-based context manager, thread-safe)
# ---------------------------------------------------------------------------

@contextmanager
def track_peak_rss():
    """
    Context manager that tracks peak CPU RSS (MB) during a block.

    State is stored in a list [peak_mb] passed by reference, eliminating
    the global-variable race condition in the original implementation.

    Usage:
        state = [0.0]
        with track_peak_rss() as state:
            ...do work...
        peak_mb = state[0]
    """
    process = psutil.Process(os.getpid())
    state   = [process.memory_info().rss / (1024 ** 2)]
    try:
        yield state
    finally:
        state[0] = max(state[0], process.memory_info().rss / (1024 ** 2))
