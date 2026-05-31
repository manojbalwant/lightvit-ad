# Jetson Nano Setup Guide

Complete setup guide for deploying LightViT-AD on NVIDIA Jetson Nano Developer Kit.

## Hardware

- **Board**: NVIDIA Jetson Nano Developer Kit (B01 or A02 revision)
- **Software**: JetPack 4.6 (L4T 32.7.x)
- **SoC**: ARM Cortex-A57 (4 cores) + Maxwell GPU (128 CUDA cores)
- **RAM**: 4 GB shared LPDDR4
- **Storage**: micro-SD card ≥32 GB (Class 10 / UHS-1 recommended)

## Power Supply

Two options:

| Connector | Rated capacity | Notes |
|---|---|---|
| micro-USB | 5 V / 2 A (10 W) | Convenient but may throttle under TRT GPU load |
| Barrel jack (5.5/2.1 mm) | 5 V / 4 A (20 W) | Recommended for sustained GPU benchmarking |

**Barrel jack setup**: Set jumper J48 on the carrier board to enable the barrel
jack power path. Without J48, the board draws from micro-USB regardless of what
is plugged in.

**Power measurement note**: The on-board INA3221 sensor measures the 5 V input
rail current regardless of which connector is used. Both paths share the same
rail; reported power values are valid under either configuration.

**If using micro-USB**: Peak measured draw at TRT INT8 is ~7.76 W (77.6% of
rated 10 W). Voltage drop through cable resistance (~0.3–1.0 Ω at 1.55 A) may
trigger PMIC under-voltage throttling, reducing CPU/GPU frequency. If throttling
occurs, FPS measurements will be conservative lower bounds. Verify:

```bash
watch -n 1 'cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq && \
             cat /sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq'
# Expect: 1479000 (CPU MHz) and 921600000 (GPU Hz) when not throttling
```

## JetPack Installation

1. Download and flash JetPack 4.6 using NVIDIA SDK Manager or Etcher.
2. Complete first-boot setup (locale, username, timezone).
3. Verify installation:

```bash
jetson_release         # should show L4T 32.7.x and JetPack 4.6
nvcc --version         # CUDA 10.2
python3 -c "import tensorrt; print(tensorrt.__version__)"  # 8.0.x
```

## Install PyTorch for JetPack 4.6

```bash
# PyTorch 1.12 wheel for JetPack 4.6 / CUDA 10.2 / Python 3.6
pip3 install torch torchvision \
  --index-url https://developer.download.nvidia.com/compute/redist/jp/v46/pytorch/
```

## Install TensorRT Python Bindings

```bash
sudo apt-get update
sudo apt-get install -y python3-libnvinfer-dev
# Verify
python3 -c "import tensorrt; print('TRT', tensorrt.__version__)"
```

## Install Project Dependencies

```bash
# On Jetson Nano:
git clone https://github.com/manojbalwant/lightvit-ad.git
cd lightvit-ad
pip3 install -r requirements_jetson.txt
```

## Transfer Trained Checkpoint from x86 Server

```bash
# On x86 server after training:
scp experiments/Railway_Inspection/best_checkpoint.pth \
    jetson@<JETSON_IP>:~/lightvit-ad/experiments/Railway_Inspection/

# Also transfer the dataset test split (or access via NFS):
scp -r dataset/Drone-Anomaly/Railway\ Inspection/test \
    jetson@<JETSON_IP>:~/lightvit-ad/dataset/Drone-Anomaly/Railway\ Inspection/
```

## Run Deployment Benchmarking

```bash
# On Jetson Nano (via SSH or locally):
cd ~/lightvit-ad
python3 scripts/deploy_jetson.py \
    --config    configs/drone_anomaly.yaml \
    --scene     "Railway Inspection" \
    --checkpoint experiments/Railway_Inspection/best_checkpoint.pth \
    --variants  base fp16 trt_fp16 trt_int8
```

TRT engine build (first run only) takes approximately 3–10 minutes on Jetson Nano.
Subsequent runs reuse the cached `.trt` engine file.

## Headless / SSH Operation

The Jetson Nano can be operated headless via SSH. The SSH channel provides
only a terminal connection and contributes no computation to any measured metric:
all inference, timing, memory sampling, and power/thermal logging run natively
on the Jetson Nano.

```bash
ssh jetson@<JETSON_IP>
# Then run the deployment script as above
```

## Verify No Throttling

Run this in a separate SSH session during benchmarking:

```bash
# Log CPU and GPU frequencies every 1 second
while true; do
  echo -n "$(date +%H:%M:%S) CPU:"
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
  echo -n " GPU:"
  cat /sys/devices/57000000.gpu/devfreq/57000000.gpu/cur_freq
  sleep 1
done
```

Expected values (no throttling, MAXN mode):
- CPU: 1479000 (1479 MHz)
- GPU: 921600000 (921 MHz)

If frequencies drop during TRT benchmarks, throttling has occurred; reported
FPS values are conservative lower bounds.

## Memory Management

The Jetson Nano has 4 GB shared RAM. At TRT INT8 evaluation, only ~80 MB
may be available. The fork-free `TegrastatsLogger` in `deploy_jetson.py`
uses `threading.Thread` (zero additional RAM) rather than `subprocess.Popen`
(which calls `os.fork()`, requiring the kernel to reserve space to duplicate
the parent process address space, risking `ENOMEM`).

If you encounter memory issues, reduce `bench_trials` in the config YAML.
