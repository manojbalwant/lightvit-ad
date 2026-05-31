#!/usr/bin/env python3
"""
scripts/deploy_jetson.py
========================
Jetson Nano four-variant deployment, accuracy evaluation, and benchmarking.

Target: NVIDIA Jetson Nano Developer Kit
        JetPack 4.6 | ARM Cortex-A57 | Maxwell GPU | 4 GB shared LPDDR4

Deployment variants (all derived from one trained FP32 checkpoint):
  base      — PyTorch FP32, ARM Cortex-A57 CPU
  fp16      — PyTorch FP16, Maxwell CUDA GPU (128 CUDA cores)
  trt_fp16  — TensorRT FP16 engine (.trt), cuDNN FP16 GEMM on Maxwell
  trt_int8  — TensorRT INT8 engine (.trt), IInt8EntropyCalibrator2

Benchmarking protocol (Section 4.3 / Table 6 of manuscript):
  • 3 independent runs × 200 within-run trials per variant
  • CUDA synchronisation before stop-clock for GPU variants
  • Sequential batch=1 for all variants (streaming deployment scenario)
  • P50/P95/P99 nearest-rank (⌈p×n⌉, IEEE/ACM standard)
  • 95% t-distribution CI (scipy.stats.t.ppf)
  • Power and thermal via sysfs INA3221 polling (fork-free threading)
    — INA3221 VDD_IN channel reads the 5V input rail regardless of
      whether the board is powered via micro-USB or barrel jack.

Usage (on Jetson Nano):
  python3 scripts/deploy_jetson.py \\
      --config    configs/drone_anomaly.yaml \\
      --scene     "Railway Inspection" \\
      --checkpoint experiments/Railway_Inspection/best_checkpoint.pth

Training is NOT performed here. The checkpoint must be trained on the
x86 server (scripts/train_x86.py) and transferred to the Jetson Nano.
"""

import os
import sys
import copy
import glob
import time
import math
import threading
import warnings
import argparse
import yaml
import types
import statistics
from contextlib import contextmanager
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from scipy import stats as scipy_stats
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, f1_score,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightvit_ad.models  import ViTTeacher, ViTStudent, CombinedModel
from lightvit_ad.dataset import DataLoader
from lightvit_ad.train   import load_checkpoint, evaluate_plot
from lightvit_ad.utils   import set_seed, track_peak_rss

# ---------------------------------------------------------------------------
# Optional imports (auto-skipped on x86 development machine)
# ---------------------------------------------------------------------------
_TRT_AVAILABLE    = False
_PYCUDA_AVAILABLE = False
_ONNX_AVAILABLE   = False
_ONNXSIM_AVAILABLE = False

try:
    import tensorrt as trt
    _TRT_AVAILABLE = True
except ImportError:
    pass

try:
    import pycuda.driver as cuda_drv
    cuda_drv.init()
    dev = cuda_drv.Device(0)
    CUDA_CTX = dev.retain_primary_context()
    CUDA_CTX.push()
    _PYCUDA_AVAILABLE = True
except (ImportError, Exception):
    pass

try:
    import onnx
    _ONNX_AVAILABLE = True
    try:
        from onnxsim import simplify as onnx_simplify
        _ONNXSIM_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass


# ===========================================================================
# Configuration
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description='LightViT-AD Jetson Nano deployment')
    p.add_argument('--config',     default='configs/drone_anomaly.yaml')
    p.add_argument('--scene',      default=None)
    p.add_argument('--checkpoint', required=True,
                   help='Path to best_checkpoint.pth from train_x86.py')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--variants',   nargs='+',
                   default=['base', 'fp16', 'trt_fp16', 'trt_int8'])
    return p.parse_args()


def load_config(path: str, scene_override: str = None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if scene_override:
        dataset_root = cfg.get('dataset_root', 'dataset/Drone-Anomaly')
        exp_root     = cfg.get('experiments_root', 'experiments')
        cfg['data_path'] = os.path.join(dataset_root, scene_override)
        cfg['save_path'] = os.path.join(
            exp_root, scene_override.replace(' ', '_'))
        cfg['valid_file'] = cfg.get('scene_valid_file', {}).get(
            scene_override, cfg.get('valid_file', 'val_seq_01'))
    # Add Jetson-specific defaults
    cfg.setdefault('bench_warmup',            20)
    cfg.setdefault('bench_trials',           200)
    cfg.setdefault('bench_independent_runs',   3)
    cfg.setdefault('bench_ci_alpha',        0.95)
    cfg.setdefault('realtime_10fps_ms',    100.0)
    cfg.setdefault('trt_workspace_gb',         1)
    cfg.setdefault('trt_calib_frames',       500)
    cfg.setdefault('trt_calib_batch_size',     1)
    cfg.setdefault('trt_calib_cache',  'trt_int8_calib.cache')
    cfg.setdefault('trt_onnx_path',    'model_trt_base.onnx')
    cfg.setdefault('trt_engine_fp16_path', 'engine_fp16.trt')
    cfg.setdefault('trt_engine_int8_path', 'engine_int8.trt')
    return types.SimpleNamespace(**cfg)


# ===========================================================================
# Power and thermal monitoring (fork-free, sysfs)
# ===========================================================================

class TegrastatsLogger:
    """
    Fork-free sysfs power and thermal logger for Jetson Nano.

    Uses threading.Thread (clone with shared memory) instead of
    subprocess.Popen (fork — may fail with ENOMEM at low RAM).

    Power measured via INA3221 VDD_IN channel (sysfs):
      /sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power0_input  (mW)
    Thermal measured by resolving zone type names to avoid hardcoded
    zone numbers (which differ between JetPack 4.6 and 5.x).

    Note on power supply: the INA3221 VDD_IN channel measures the 5V
    input rail current regardless of whether the board is powered via
    micro-USB or barrel jack; both supply paths share the same rail.
    """

    _POWER_GLOB = (
        '/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power0_input'
    )
    _THERMAL_ROOT = '/sys/class/thermal'

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s   = interval_s
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.power_mw:   List[float] = []
        self.thermal_c:  Dict[str, List[float]] = {}
        self._power_path: Optional[str]  = self._find_power_path()
        self._zone_paths: Dict[str, str] = self._resolve_thermal_zones()

    def _find_power_path(self) -> Optional[str]:
        paths = glob.glob(self.POWER_GLOB if hasattr(self, 'POWER_GLOB')
                          else self._POWER_GLOB)
        return paths[0] if paths else None

    def _resolve_thermal_zones(self) -> Dict[str, str]:
        """
        Resolve zone paths by reading type files (JetPack version agnostic).
        Returns dict: zone_name → temperature_path
        """
        zones = {}
        zone_dirs = sorted(glob.glob(
            os.path.join(self._THERMAL_ROOT, 'thermal_zone*')
        ))
        for d in zone_dirs:
            type_file = os.path.join(d, 'type')
            temp_file = os.path.join(d, 'temp')
            if os.path.exists(type_file) and os.path.exists(temp_file):
                try:
                    with open(type_file) as f:
                        zone_type = f.read().strip()
                    zones[zone_type] = temp_file
                except OSError:
                    pass
        return zones

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            # Power (mW)
            if self._power_path:
                try:
                    with open(self._power_path) as f:
                        self.power_mw.append(float(f.read().strip()))
                except (OSError, ValueError):
                    pass
            # Thermal (°C from milli-degrees)
            for name, path in self._zone_paths.items():
                try:
                    with open(path) as f:
                        val = float(f.read().strip()) / 1000.0
                    self.thermal_c.setdefault(name, []).append(val)
                except (OSError, ValueError):
                    pass
            time.sleep(self.interval_s)

    def start(self) -> None:
        self.power_mw.clear()
        self.thermal_c.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> Dict[str, float]:
        pw = self.power_mw
        gpu_temps = self.thermal_c.get('GPU-therm', [])
        cpu_temps = self.thermal_c.get('CPU-therm', [])
        return {
            'avg_power_w':    round(statistics.mean(pw) / 1000, 3) if pw else 0.0,
            'peak_power_w':   round(max(pw) / 1000, 3)             if pw else 0.0,
            'gpu_temp_steady': round(statistics.mean(gpu_temps), 2) if gpu_temps else float('nan'),
            'cpu_temp_steady': round(statistics.mean(cpu_temps), 2) if cpu_temps else float('nan'),
        }


# ===========================================================================
# Latency and throughput benchmarking
# ===========================================================================

def _nearest_rank_percentile(data: List[float], p: float) -> float:
    """P-th percentile using nearest-rank formula: index = ceil(p×n) - 1."""
    n   = len(data)
    idx = max(0, math.ceil(p * n) - 1)
    return sorted(data)[idx]


def measure_latency_extended(
    model,
    x: torch.Tensor,
    warmup: int  = 20,
    trials: int  = 200,
    cuda_device: Optional[torch.device] = None,
) -> Tuple[float, float, float, float, float, float]:
    """
    One independent run of latency benchmarking.

    Returns: mean_ms, std_ms, p50_ms, p95_ms, p99_ms, peak_rss_mb
    """
    if hasattr(model, 'eval'):
        model.eval()

    is_cuda = (cuda_device is not None)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(x)
    if is_cuda:
        torch.cuda.synchronize()

    # Trials
    latencies = []
    mem_state = [0.0]
    process   = psutil.Process(os.getpid())

    with track_peak_rss() as mem_state:
        for _ in range(trials):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    mean_ms = statistics.mean(latencies)
    std_ms  = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    p50     = _nearest_rank_percentile(latencies, 0.50)
    p95     = _nearest_rank_percentile(latencies, 0.95)
    p99     = _nearest_rank_percentile(latencies, 0.99)
    return mean_ms, std_ms, p50, p95, p99, mem_state[0]


def measure_latency_multi_run(
    model,
    x: torch.Tensor,
    n_runs: int,
    warmup: int,
    trials: int,
    ci_alpha: float = 0.95,
    cuda_device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Multi-run latency: n_runs × trials, 95% t-distribution CI.
    Returns aggregated statistics across all independent runs.
    """
    run_means, run_p95s, run_rss = [], [], []

    for _ in range(n_runs):
        m, s, p50, p95, p99, rss = measure_latency_extended(
            model, x, warmup, trials, cuda_device
        )
        run_means.append(m)
        run_p95s.append(p95)
        run_rss.append(rss)

    n      = len(run_means)
    mu     = statistics.mean(run_means)
    sigma  = statistics.stdev(run_means) if n > 1 else 0.0
    t_crit = scipy_stats.t.ppf((1 + ci_alpha) / 2, df=max(n - 1, 1))
    ci_hw  = t_crit * sigma / math.sqrt(n)

    return {
        'mean_ms':       round(mu,            3),
        'std_ms':        round(sigma,          3),
        'ci_hw_ms':      round(ci_hw,          3),
        'p95_ms':        round(statistics.mean(run_p95s), 3),
        'peak_rss_mb':   round(max(run_rss),   1),
    }


def measure_throughput(
    model,
    x: torch.Tensor,
    warmup: int   = 20,
    trials: int   = 100,
    cuda_device:  Optional[torch.device] = None,
) -> Dict[str, float]:
    """Sequential batch=1 throughput (FPS)."""
    if hasattr(model, 'eval'):
        model.eval()
    is_cuda = (cuda_device is not None)

    for _ in range(warmup):
        with torch.no_grad():
            model(x)
    if is_cuda:
        torch.cuda.synchronize()

    fps_list = []
    for _ in range(trials):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        fps_list.append(1.0 / max(t1 - t0, 1e-9))

    mean_fps = statistics.mean(fps_list)
    std_fps  = statistics.stdev(fps_list) if len(fps_list) > 1 else 0.0
    return {'mean_fps': round(mean_fps, 3), 'std_fps': round(std_fps, 3)}


# ===========================================================================
# TensorRT pipeline  (skipped on non-Jetson hosts)
# ===========================================================================

def _sanitize_int64(model_path: str) -> None:
    """Convert INT64 initializers/constants to INT32 for TRT 7/8 compatibility."""
    if not _ONNX_AVAILABLE:
        return
    model = onnx.load(model_path)
    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.INT64:
            arr = np.array(onnx.numpy_helper.to_array(init), dtype=np.int32)
            new = onnx.numpy_helper.from_array(arr, name=init.name)
            model.graph.initializer.remove(init)
            model.graph.initializer.append(new)
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.t.data_type == onnx.TensorProto.INT64:
                    arr = np.frombuffer(attr.t.raw_data, dtype=np.int64)
                    attr.t.raw_data = arr.astype(np.int32).tobytes()
                    attr.t.data_type = onnx.TensorProto.INT32
    onnx.save(model, model_path)


def export_onnx(model, config, save_path: str, opset: int = 11) -> str:
    """Export combined model to ONNX (opset 11 for TRT 8.x compatibility)."""
    onnx_path = os.path.join(save_path, config.trt_onnx_path)
    dummy     = torch.randn(1, 3, config.image_size, config.image_size)
    model.eval()
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=opset,
        input_names=['input'],
        output_names=['cls_token', 'dist_token'],
        dynamic_axes=None,  # static batch=1 for Jetson Nano
        do_constant_folding=True,
        verbose=False,
    )
    _sanitize_int64(onnx_path)
    if _ONNXSIM_AVAILABLE:
        m, ok = onnx_simplify(onnx.load(onnx_path))
        if ok:
            onnx.save(m, onnx_path)
    return onnx_path


def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    config,
    use_int8: bool = False,
    calib_loader=None,
) -> Optional[str]:
    """
    Build a TensorRT engine from an ONNX graph.
    Returns engine_path on success, None if TRT is unavailable.
    """
    if not _TRT_AVAILABLE:
        print('[WARN] TensorRT not available — skipping engine build.')
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(
             1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
         ) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            config.trt_workspace_gb * (1 << 30)
        )
        cfg.set_flag(trt.BuilderFlag.FP16)
        if use_int8 and calib_loader is not None:
            cfg.set_flag(trt.BuilderFlag.INT8)
            cfg.int8_calibrator = calib_loader

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f'  ONNX parse error: {parser.get_error(i)}')
                return None

        serialized = builder.build_serialized_network(network, cfg)
        if serialized is None:
            print('[ERROR] TRT engine build failed.')
            return None

        with open(engine_path, 'wb') as f:
            f.write(serialized)
    return engine_path


class TRTInferenceSession:
    """
    Duck-typed nn.Module interface for TensorRT engine inference.
    Supports model.eval() and model(x) call convention.
    """
    def __init__(self, engine_path: str) -> None:
        if not (_TRT_AVAILABLE and _PYCUDA_AVAILABLE):
            raise RuntimeError('TensorRT or pycuda not available.')
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime    = trt.Runtime(TRT_LOGGER)
        with open(engine_path, 'rb') as f:
            self.engine  = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # Allocate buffers
        self._allocate_buffers()

    def _allocate_buffers(self):
        import pycuda.driver as cuda_drv
        self._bindings = []
        self._host_bufs   = {}
        self._device_bufs = {}
        for binding in self.engine:
            shape = tuple(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem   = cuda_drv.pagelocked_empty(shape, dtype)
            device_mem = cuda_drv.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(device_mem))
            self._host_bufs[binding]   = host_mem
            self._device_bufs[binding] = device_mem
        self._stream = cuda_drv.Stream()

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor):
        import pycuda.driver as cuda_drv
        inp_name = self.engine.get_binding_name(0)
        np.copyto(self._host_bufs[inp_name],
                  x.cpu().numpy().astype(np.float32).ravel())
        cuda_drv.memcpy_htod_async(
            self._device_bufs[inp_name],
            self._host_bufs[inp_name],
            self._stream,
        )
        self.context.execute_async_v2(self._bindings, self._stream.handle)
        outputs = []
        for i in range(1, self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            cuda_drv.memcpy_dtoh_async(
                self._host_bufs[name],
                self._device_bufs[name],
                self._stream,
            )
            self._stream.synchronize()
            outputs.append(
                torch.from_numpy(self._host_bufs[name].copy())
            )
        return tuple(outputs)


# ===========================================================================
# Variant runners
# ===========================================================================

def _build_variant(
    variant: str,
    teacher_base: ViTTeacher,
    student_base: ViTStudent,
    config,
    mean,
    std,
    cpu_device: torch.device,
    cuda_device: Optional[torch.device],
    save_path: str,
) -> Tuple[object, object, Optional[torch.device]]:
    """
    Build the (teacher, student_or_trt_session, inference_device) triple
    for a given variant.
    """
    if variant == 'base':
        t = copy.deepcopy(teacher_base).to(cpu_device)
        s = copy.deepcopy(student_base).to(cpu_device)
        for p in t.parameters(): p.requires_grad_(False)
        for p in s.parameters(): p.requires_grad_(False)
        return t, s, None   # None → CPU inference, no CUDA sync

    elif variant == 'fp16':
        if cuda_device is None:
            print('[WARN] fp16 variant requires CUDA — skipping.')
            return None, None, None
        t = copy.deepcopy(teacher_base).half().to(cuda_device)
        s = copy.deepcopy(student_base).half().to(cuda_device)
        for p in t.parameters(): p.requires_grad_(False)
        for p in s.parameters(): p.requires_grad_(False)
        return t, s, cuda_device

    elif variant in ('trt_fp16', 'trt_int8'):
        if not (_TRT_AVAILABLE and _PYCUDA_AVAILABLE and _ONNX_AVAILABLE):
            print(f'[WARN] {variant}: TRT/pycuda/ONNX unavailable — skipping.')
            return None, None, None

        # Build or load TRT engine
        engine_name = (config.trt_engine_fp16_path if variant == 'trt_fp16'
                       else config.trt_engine_int8_path)
        engine_path = os.path.join(save_path, engine_name)

        if not os.path.exists(engine_path):
            print(f'  Building TRT engine for {variant}...')
            combined_fp32 = CombinedModel(
                copy.deepcopy(teacher_base).to(cpu_device),
                copy.deepcopy(student_base).to(cpu_device),
            ).eval()
            onnx_path = export_onnx(combined_fp32, config, save_path)

            calib = None
            if variant == 'trt_int8':
                calib = _build_int8_calibrator(config, mean, std, save_path)

            engine_path = build_trt_engine(
                onnx_path, engine_path, config,
                use_int8=(variant == 'trt_int8'),
                calib_loader=calib,
            )
            if engine_path is None:
                return None, None, None

        trt_session = TRTInferenceSession(engine_path)
        # Teacher runs in FP32 on CPU for anomaly score computation
        t = copy.deepcopy(teacher_base).to(cpu_device)
        for p in t.parameters(): p.requires_grad_(False)
        return t, trt_session, None   # No separate CUDA sync needed (TRT handles it)

    else:
        raise ValueError(f'Unknown variant: {variant}')


def _build_int8_calibrator(config, mean, std, save_path: str):
    """Build IInt8EntropyCalibrator2 from training normal frames."""
    if not _TRT_AVAILABLE:
        return None

    train_folder = os.path.join(config.data_path, 'train', 'frames')
    calib_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    calib_dataset = DataLoader(
        train_folder, calib_transform,
        config.image_size, config.image_size,
        time_step=config.num_frames,
    )
    n = min(config.trt_calib_frames, len(calib_dataset))
    calib_subset = data.Subset(calib_dataset, list(range(n)))
    calib_loader = data.DataLoader(calib_subset, batch_size=1, shuffle=False)

    class _Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, loader, cache_path):
            super().__init__()
            import pycuda.driver as cuda_drv
            self._iter   = iter(loader)
            self._cache  = cache_path
            self._buf    = cuda_drv.mem_alloc(
                1 * 3 * config.image_size * config.image_size * 4
            )

        def get_batch_size(self): return 1

        def get_batch(self, names):
            try:
                batch = next(self._iter)
                img   = batch['standard'][:, 0].float().numpy()
                img   = np.ascontiguousarray(img)
                import pycuda.driver as cuda_drv
                cuda_drv.memcpy_htod(self._buf, img)
                return [int(self._buf)]
            except StopIteration:
                return None

        def read_calibration_cache(self):
            if os.path.exists(self._cache):
                with open(self._cache, 'rb') as f: return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self._cache, 'wb') as f: f.write(cache)

    cache_path = os.path.join(save_path, config.trt_calib_cache)
    return _Calibrator(calib_loader, cache_path)


# ===========================================================================
# Main
# ===========================================================================

def main():
    args   = parse_args()
    config = load_config(args.config, args.scene)
    set_seed(args.seed)

    os.makedirs(config.save_path, exist_ok=True)
    cpu_device  = torch.device('cpu')
    cuda_device = torch.device('cuda:0') if torch.cuda.is_available() else None

    print(f'\nJetson Nano deployment benchmarking')
    print(f'  Checkpoint: {args.checkpoint}')
    print(f'  Scene:      {config.data_path}')
    print(f'  Variants:   {args.variants}')
    print(f'  TRT:        {_TRT_AVAILABLE}  pycuda: {_PYCUDA_AVAILABLE}')

    # ── Load checkpoint (FIX-A: expect teacher_state_dict + student_state_dict)
    teacher_sd, student_sd, mean_std, best_auc = load_checkpoint(
        args.checkpoint, cpu_device
    )
    mean = mean_std['mean']
    std  = mean_std['std']

    teacher_base = ViTTeacher(pretrained=False).to(cpu_device)
    student_base = ViTStudent(pretrained=False).to(cpu_device)
    teacher_base.load_state_dict(teacher_sd)
    student_base.load_state_dict(student_sd)
    teacher_base.eval()
    student_base.eval()

    print(f'  Checkpoint loaded. Best training AUC: {best_auc:.4f}')

    # ── Benchmark each variant ───────────────────────────────────────────────
    all_results = []
    bench_input_cpu  = torch.rand(1, 3, config.image_size, config.image_size)
    bench_input_cuda = (bench_input_cpu.to(cuda_device)
                        if cuda_device else None)

    for variant in args.variants:
        print(f'\n{"="*60}')
        print(f'  Variant: {variant}')
        print(f'{"="*60}')

        t, s, inf_device = _build_variant(
            variant, teacher_base, student_base,
            config, mean, std,
            cpu_device, cuda_device, config.save_path,
        )
        if t is None:
            print(f'  Skipped.')
            continue

        # Combine for benchmarking
        if isinstance(s, TRTInferenceSession):
            bench_model = s
            bench_x     = bench_input_cpu
        elif inf_device is not None:   # fp16 / GPU
            bench_model = CombinedModel(t, s)
            bench_x     = bench_input_cuda.half()
        else:                          # base / CPU
            bench_model = CombinedModel(t, s)
            bench_x     = bench_input_cpu

        # 1. Accuracy evaluation
        print('  Running accuracy evaluation...')
        acc = evaluate_plot(
            t, s if not isinstance(s, TRTInferenceSession) else student_base,
            config, mean, std,
            variant=variant, device=cpu_device,
        )

        # 2. Power/thermal logger
        logger = TegrastatsLogger(interval_s=0.5)
        logger.start()
        time.sleep(2.0)   # 2-second thermal pre-soak

        # 3. Latency (multi-run)
        torch.set_num_threads(
            psutil.cpu_count(logical=False) or 4
        )
        print('  Benchmarking latency...')
        lat = measure_latency_multi_run(
            bench_model, bench_x,
            n_runs   = config.bench_independent_runs,
            warmup   = config.bench_warmup,
            trials   = config.bench_trials,
            ci_alpha = config.bench_ci_alpha,
            cuda_device = inf_device,
        )

        # 4. Throughput
        print('  Benchmarking throughput...')
        thr = measure_throughput(
            bench_model, bench_x,
            warmup = config.bench_warmup,
            trials = 100,
            cuda_device = inf_device,
        )

        logger.stop()
        time.sleep(1.0)
        thermal = logger.summary()

        # 5. Memory
        gpu_alloc_mb = (
            torch.cuda.max_memory_allocated(inf_device) / (1024 ** 2)
            if inf_device is not None else 0.0
        )
        if inf_device is not None:
            torch.cuda.reset_peak_memory_stats(inf_device)
        ram_avail_gb = psutil.virtual_memory().available / (1024 ** 3)

        # 6. Energy per frame
        j_per_frame = (thermal['avg_power_w'] / thr['mean_fps']
                       if thr['mean_fps'] > 0 else float('nan'))

        row = {
            'variant':          variant,
            # Accuracy
            'auc':              acc.get('auc'),
            'pr_auc':           acc.get('avg_precision'),
            'best_f1':          acc.get('best_f1'),
            'eer':              acc.get('eer'),
            # Latency
            'mean_lat_ms':      lat['mean_ms'],
            'std_lat_ms':       lat['std_ms'],
            'ci_hw_ms':         lat['ci_hw_ms'],
            'p95_ms':           lat['p95_ms'],
            # Real-time criterion
            'realtime_10fps':   lat['p95_ms'] < config.realtime_10fps_ms,
            # Throughput
            'mean_fps':         thr['mean_fps'],
            'std_fps':          thr['std_fps'],
            # Memory
            'peak_rss_mb':      lat['peak_rss_mb'],
            'gpu_alloc_mb':     round(gpu_alloc_mb, 1),
            'ram_avail_gb':     round(ram_avail_gb, 3),
            # Power / thermal
            'avg_power_w':      thermal['avg_power_w'],
            'peak_power_w':     thermal['peak_power_w'],
            'gpu_temp_c':       thermal['gpu_temp_steady'],
            'cpu_temp_c':       thermal['cpu_temp_steady'],
            # Energy
            'j_per_frame':      round(j_per_frame, 3),
        }
        all_results.append(row)
        print(f'\n  {variant}: AUC={row["auc"]}  FPS={row["mean_fps"]}  '
              f'P95={row["p95_ms"]}ms  {row["avg_power_w"]}W  '
              f'{row["j_per_frame"]} J/frame')

    # ── Results table ────────────────────────────────────────────────────────
    if all_results:
        df = pd.DataFrame(all_results).set_index('variant')
        print('\n' + '='*80)
        print(df.T.to_string())
        print('='*80)
        csv_path = os.path.join(config.save_path, 'jetson_results.csv')
        df.to_csv(csv_path)
        print(f'\nResults saved: {csv_path}')


if __name__ == '__main__':
    main()
