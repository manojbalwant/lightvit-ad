#!/usr/bin/env python3
"""
scripts/train_x86.py
====================
Training entry point for the x86 GPU server.

Platform (used in paper):
  IIT Patna server, x86_64, 24 physical cores (48 threads), 252 GB RAM,
  Ubuntu kernel 4.4.0-87-generic, PyTorch v1.12.1, CUDA 11.x.

Usage:
  python scripts/train_x86.py \\
      --config configs/drone_anomaly.yaml \\
      --scene  "Railway Inspection" \\
      --device cuda:0

What this script does:
  1. Computes per-scene mean/std from training normal frames.
  2. Trains the student for config.joint_epochs (default 15) using
     OneCycleLR + AdamW + mixed-precision (torch.cuda.amp).
  3. Saves the BEST-VAL-AUC epoch checkpoint via copy.deepcopy (FIX-B).
     Checkpoint keys: teacher_state_dict, student_state_dict, mean_std, best_auc.
     These keys are required by deploy_jetson.py.
  4. Runs evaluate_plot() and reports ROC-AUC, PR-AUC, Best-F1, EER.

The trained checkpoint is subsequently transferred to the Jetson Nano
for four-variant deployment benchmarking (see scripts/deploy_jetson.py).
"""

import os
import sys
import copy
import argparse
import yaml
import types

import torch
import torch.utils.data as data
from torchvision import transforms

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightvit_ad.models  import ViTTeacher, ViTStudent
from lightvit_ad.dataset import DataLoader, compute_mean_and_std
from lightvit_ad.train   import (
    train_epoch, validate, evaluate_plot, save_checkpoint
)
from lightvit_ad.utils   import set_seed, compute_flops_and_params


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='LightViT-AD training (x86)')
    p.add_argument('--config', default='configs/drone_anomaly.yaml')
    p.add_argument('--scene',  default=None,
                   help='Scene name (overrides config data_path/save_path)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed',   type=int, default=42)
    return p.parse_args()


def load_config(path: str, scene_override: str = None):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if scene_override:
        dataset_root = cfg.get('dataset_root', 'dataset/Drone-Anomaly')
        exp_root     = cfg.get('experiments_root', 'experiments')
        cfg['data_path']  = os.path.join(dataset_root, scene_override)
        cfg['save_path']  = os.path.join(exp_root, scene_override.replace(' ', '_'))
        cfg['valid_file'] = cfg.get('scene_valid_file', {}).get(
            scene_override, cfg.get('valid_file', 'val_seq_01')
        )

    # Convert to namespace for attribute access
    ns = types.SimpleNamespace(**cfg)
    return ns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    config = load_config(args.config, args.scene)
    set_seed(args.seed)

    device     = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cpu_device = torch.device('cpu')

    print(f'\n{"="*60}')
    print(f'  LightViT-AD training')
    print(f'  Scene     : {config.data_path}')
    print(f'  Save to   : {config.save_path}')
    print(f'  Device    : {device}')
    print(f'{"="*60}\n')

    os.makedirs(config.save_path, exist_ok=True)

    # ── Mean / Std ──────────────────────────────────────────────────────────
    train_folder = os.path.join(config.data_path, 'train', 'frames')
    mean, std    = compute_mean_and_std(
        train_folder, config.image_size, config.image_size,
        device=device, batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    mean_std = {'mean': mean, 'std': std}
    print(f'Mean: {mean}  Std: {std}')

    # ── Data loaders ────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.RandomResizedCrop(config.image_size)], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        ], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = DataLoader(
        train_folder, train_transform,
        config.image_size, config.image_size,
        time_step=config.num_frames,
    )
    train_loader = data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )

    valid_folder  = os.path.join(config.data_path, 'test', 'frames',
                                 config.valid_file)
    valid_dataset = DataLoader(
        valid_folder, test_transform,
        config.image_size, config.image_size,
        time_step=config.num_frames,
    )
    valid_loader = data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )

    # ── Models ──────────────────────────────────────────────────────────────
    teacher = ViTTeacher(pretrained=True).to(device)
    student = ViTStudent(pretrained=False).to(device)

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── Optimiser / scheduler / scaler ──────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.lr, weight_decay=config.wd
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.lr,
        steps_per_epoch=len(train_loader),
        epochs=config.joint_epochs,
    )
    scaler = torch.cuda.amp.GradScaler()

    # ── Training loop (FIX-B: best-epoch checkpoint) ─────────────────────
    best_auc          = 0.0
    best_student_state = None

    for epoch in range(1, config.joint_epochs + 1):
        train_loss          = train_epoch(
            teacher, student, train_loader, optimizer, scheduler, scaler,
            config, device,
        )
        val_loss, val_auc   = validate(teacher, student, valid_loader, config, device)

        print(f'Epoch {epoch:2d}/{config.joint_epochs}  '
              f'train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  '
              f'val_auc={val_auc:.4f}')

        # FIX-B: track and deepcopy the best-val-AUC student state
        if val_auc > best_auc:
            best_auc           = val_auc
            best_student_state = copy.deepcopy(student.state_dict())
            print(f'  *** New best AUC: {best_auc:.4f} at epoch {epoch} ***')

    # Restore best student before saving / evaluation
    student.load_state_dict(best_student_state)

    ckpt_path = os.path.join(config.save_path, 'best_checkpoint.pth')
    save_checkpoint(teacher, student, mean_std, best_auc, ckpt_path)
    print(f'\nCheckpoint saved: {ckpt_path}')

    # ── Move to CPU for evaluation ──────────────────────────────────────────
    teacher.to(cpu_device)
    student.to(cpu_device)

    # ── Evaluate ────────────────────────────────────────────────────────────
    print('\nRunning full test-set evaluation (base FP32)...')
    metrics = evaluate_plot(
        teacher, student, config,
        mean, std, variant='base', device=cpu_device,
    )
    print(f'\nResults: {metrics}')

    # ── FLOPs / param count ─────────────────────────────────────────────────
    from lightvit_ad.models import CombinedModel
    combined = CombinedModel(teacher, student)
    example  = torch.rand(1, 3, config.image_size, config.image_size)
    gflops, params_M = compute_flops_and_params(combined, example)
    print(f'\nCombined model: {params_M:.3f} M params, {gflops:.3f} GFLOPs')
    print('\nTraining complete.')


if __name__ == '__main__':
    main()
