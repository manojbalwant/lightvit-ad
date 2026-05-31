# LightViT-AD

**LightViT-AD: A Dynamic Post-Training Quantized Vision Transformer for Unsupervised Anomaly Detection in UAV Imagery**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 1.12](https://img.shields.io/badge/PyTorch-1.12-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-preprint-red.svg)](#citation)

> Manoj Kumar Balwant, Shivendu Mishra, Rajiv Misra  
> Department of Computer Science and Engineering, IIT Patna, India

---

## Overview

LightViT-AD is a lightweight Vision Transformer (ViT) teacher–student framework for
**one-class unsupervised anomaly detection** in aerial video, trained exclusively on
normal frames. A frozen pretrained DeiT-tiny teacher produces CLS and DIST tokens
fused into a 192-dim latent; a depth-reduced student (6 blocks) reproduces these
tokens via MSE loss **without ever processing raw image pixels**.

At inference, the per-frame anomaly score is the MSE between teacher and student
tokens — a scalar computed from two 192-dimensional vectors.

**Deployment**: The trained FP32 checkpoint is deployed on NVIDIA Jetson Nano
(JetPack 4.6) in four configurations:

| Variant | Backend | FPS (Jetson Nano) | J/frame |
|---|---|---|---|
| base | PyTorch FP32, ARM CPU | 1.54 | 3.64 |
| fp16 | PyTorch FP16, Maxwell GPU | 17.72 | 0.39 |
| trt_fp16 | TensorRT FP16 | 24.40 | 0.31 |
| trt_int8 | TensorRT INT8 (entropy-calibrated) | 24.12 | 0.31 |

All three GPU variants meet the ≥10 FPS real-time criterion within the 10 W UAV power budget.

---

## Repository Structure

```
lightvit-ad/
├── lightvit_ad/
│   ├── __init__.py
│   ├── models.py          # ViTTeacher, ViTStudent, CombinedModel
│   ├── dataset.py         # DataLoader, compute_mean_and_std
│   ├── train.py           # Training loop, validation, evaluate_plot
│   └── utils.py           # Seeding, weight init, FLOPs profiling
├── configs/
│   ├── drone_anomaly.yaml # Drone-Anomaly dataset config
│   └── uit_adrone.yaml    # UIT-ADrone dataset config
├── scripts/
│   ├── train_x86.py       # Training on x86 GPU server
│   └── deploy_jetson.py   # Jetson Nano 4-variant benchmarking
├── docs/
│   └── jetson_setup.md    # Jetson Nano environment setup guide
├── experiments/           # Checkpoints and results (git-ignored)
├── dataset/               # Datasets (git-ignored)
├── requirements.txt       # x86 server dependencies
├── requirements_jetson.txt# Jetson Nano dependencies
└── README.md
```

---

## Datasets

### Drone-Anomaly
Download from the [official repository](https://github.com/jinpujin/Drone-Anomaly).
Place under `dataset/Drone-Anomaly/` with scene subfolders:
```
dataset/Drone-Anomaly/
├── Highway/
│   ├── train/frames/
│   └── test/
│       ├── frames/
│       └── test_frame_mask/
├── Railway Inspection/
├── Solar Panel Inspection/
...
```

### UIT-ADrone
Download from [UIT-ADrone](https://sites.google.com/uit.edu.vn/uit-adrone).
Place under `dataset/UIT-ADrone/`.

---

## Installation

### x86 Server (Training)

```bash
git clone https://github.com/manojbalwant/lightvit-ad.git
cd lightvit-ad
pip install -r requirements.txt
```

### Jetson Nano (Deployment & Benchmarking)

Follow the detailed setup guide in [`docs/jetson_setup.md`](docs/jetson_setup.md),
then:

```bash
pip3 install -r requirements_jetson.txt
```

**Power supply note**: For sustained GPU benchmarking, use a regulated
5 V / 4 A barrel jack supply (20 W rated) with jumper J48 installed.
If using micro-USB (5 V / 2 A, 10 W rated), CPU/GPU frequency throttling
may occur under TRT load; measured FPS values will be conservative lower bounds.
Verify by logging `/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq`
and `/sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq` during benchmarks.

---

## Training (x86 Server)

```bash
# Edit configs/drone_anomaly.yaml to set data_path, save_path, valid_file
python scripts/train_x86.py --config configs/drone_anomaly.yaml --scene "Railway Inspection"
```

**What this does:**
1. Computes per-scene mean/std from training normal frames.
2. Trains the student for 15 epochs with OneCycleLR + AdamW + mixed precision.
3. Saves the **best-val-AUC epoch** checkpoint (teacher + student + mean_std) to
   `experiments/<scene>/best_checkpoint.pth`.
4. Runs `evaluate_plot` on the test set and saves `score_plot_base.png`.
5. Reports ROC-AUC, PR-AUC, Best-F1, EER, and EER threshold.

**Training platform used in the paper:**  
IIT Patna server, x86_64, 24 physical cores (48 threads), 252 GB RAM,
Ubuntu (kernel 4.4.0-87-generic), PyTorch v1.12.1, CUDA 11.x.

---

## Jetson Nano Deployment & Benchmarking

Transfer the trained checkpoint to the Jetson Nano, then:

```bash
# On the Jetson Nano (via SSH or locally):
python3 scripts/deploy_jetson.py \
    --config configs/drone_anomaly.yaml \
    --scene "Railway Inspection" \
    --checkpoint experiments/RailwayInspection/best_checkpoint.pth
```

**What this does:**
1. Loads the FP32 checkpoint trained on the x86 server.
2. Builds all four deployment variants (FP32/FP16/TRT-FP16/TRT-INT8).
3. Runs anomaly detection accuracy evaluation (ROC-AUC, PR-AUC, F1, EER)
   for all variants.
4. Benchmarks latency (3 runs × 200 trials, P50/P95/P99, 95% CI),
   throughput (sequential batch=1), memory (CPU RSS + GPU VRAM),
   power, and thermal (INA3221 sysfs polling).
5. Prints and saves a results table.

**Experimental setup (paper):**  
Jetson Nano Developer Kit (JetPack 4.6, Maxwell GPU 128 CUDA cores,
ARM Cortex-A57, 4 GB shared LPDDR4), operated in headless mode via SSH.
Power measured via INA3221 VDD_IN channel (source-agnostic: valid for
both micro-USB and barrel jack inputs).

## Results

### Drone-Anomaly (FP32 base / TRT INT8)

| Scene | AUC (FP32) | AUC (TRT INT8) | PR-AUC (FP32) | Best-F1 (FP32) | EER (FP32) |
|---|---|---|---|---|---|
| Highway | 0.890 | 0.870 | 0.784 | 0.794 | 0.243 |
| Farmland | 0.895 | 0.893 | 0.411 | 0.831 | 0.172 |
| Solar Panel | 0.921 | 0.918 | 0.873 | 0.829 | 0.183 |
| Bike Roundabout | 0.737 | **0.755** | 0.855 | 0.690 | 0.334 |
| Railway | 0.713 | 0.685 | 0.161 | 0.748 | 0.378 |

### UIT-ADrone

| Model | AUC | EER |
|---|---|---|
| LightViT-AD (FP32) | 0.7182 | 0.345 |
| LightViT-AD (PTQ INT8) | 0.6990 | 0.360 |

### Jetson Nano (key metrics)

| Variant | Mean latency (ms) | FPS | Avg power (W) | J/frame |
|---|---|---|---|---|
| base (FP32 CPU) | 693.3 | 1.54 | 5.60 | 3.636 |
| fp16 (Maxwell GPU) | 56.4 | 17.72 | 6.89 | 0.389 |
| trt_fp16 | 41.0 | 24.40 | 7.49 | 0.307 |
| trt_int8 | 41.4 | 24.12 | 7.42 | 0.308 |

---

## Citation

If you use this code, please cite:

```bibtex
@article{balwant2025lightvitad,
  title   = {{LightViT-AD}: Lightweight Vision Transformer Distillation for Unsupervised UAV Anomaly Detection with Real-Time Edge Inference},
  author  = {Balwant, Manoj Kumar and Mishra, Shivendu and Misra, Rajiv},
  journal = {Aerospace Science and Technology},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).
