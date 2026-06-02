"""
lightvit_ad/train.py
====================
Training loop, validation, and evaluation with all fixes applied.

Fixes over the original script
-------------------------------
FIX-A  Checkpoint format: saves both teacher_state_dict and student_state_dict
       (original saved only {'state_dict': student.state_dict()}).
       File 2 (jetson_anomaly_detection_v3) expects teacher_state_dict and
       student_state_dict; the original key caused KeyError on load.

FIX-B  validate() now returns both val_loss and val_auc so best-epoch
       tracking is possible.

FIX-C  evaluate_plot() now additionally reports PR-AUC, Best-F1, and the
       corresponding decision threshold, addressing Reviewer R3.4.
"""

import os
import copy
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, f1_score,
)
import matplotlib
matplotlib.use('Agg')   # headless-safe
import matplotlib.pyplot as plt

from .dataset import DataLoader
from .models import ViTTeacher, ViTStudent


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    teacher: ViTTeacher,
    student: ViTStudent,
    loader,
    optimizer,
    scheduler,
    scaler,
    config,
    device: torch.device,
) -> float:
    """One training epoch. Teacher is frozen; student is updated."""
    teacher.eval()
    student.train()
    epoch_loss = 0.0

    for batch in tqdm(loader, desc='Training', leave=False):
        inputs = batch['standard'].to(device, non_blocking=True)
        inputs = inputs[:, 0].float()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                t_tokens, latent = teacher(inputs)
            s_tokens = student(latent)
            loss = (
                F.mse_loss(s_tokens[0], t_tokens[0]) +
                F.mse_loss(s_tokens[1], t_tokens[1])
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        epoch_loss += loss.item()

    return epoch_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Validation  (FIX-C: returns val_auc for best-epoch tracking)
# ---------------------------------------------------------------------------

def validate(
    teacher: ViTTeacher,
    student: ViTStudent,
    loader,
    config,
    device: torch.device,
):
    """
    Run validation; return (mean_loss, roc_auc).

    Labels are loaded from the .npy mask file corresponding to config.valid_file.
    """
    teacher.eval()
    student.eval()
    losses = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch['standard'].to(device)
            inputs = inputs[:, 0].float()
            t_tokens, latent = teacher(inputs)
            s_tokens = student(latent)
            loss = (
                F.mse_loss(s_tokens[0], t_tokens[0]) +
                F.mse_loss(s_tokens[1], t_tokens[1])
            )
            losses.append(loss.item())

    label_path  = os.path.join(config.data_path, 'test', 'test_frame_mask',
                               config.valid_file + '.npy')
    true_labels = np.load(label_path)
    n = min(len(true_labels), len(losses))
    auc = roc_auc_score(true_labels[:n], losses[:n])
    return float(np.mean(losses)), float(auc)


# ---------------------------------------------------------------------------
# Full evaluation with plotting (FIX-D: PR-AUC, Best-F1, threshold)
# ---------------------------------------------------------------------------

def evaluate_plot(
    teacher: ViTTeacher,
    student: ViTStudent,
    config,
    mean,
    std,
    variant: str,
    device: torch.device,
):
    """
    Evaluate on the full test set, plot anomaly scores, and return metrics.

    Returns:
        dict with keys: auc, eer, avg_precision, best_f1, best_threshold
    """
    test_path  = os.path.join(config.data_path, 'test')
    label_path = os.path.join(test_path, 'test_frame_mask')
    teacher.eval()
    student.eval()

    path_scenes = sorted(glob.glob(os.path.join(test_path, 'frames', '*')))
    all_losses: list = []
    all_labels: list = []

    for idx_video, path_scene in enumerate(path_scenes):
        scene_name = os.path.basename(path_scene)
        print(f'  [{idx_video+1}/{len(path_scenes)}] {scene_name}')

        test_dataset = DataLoader(
            path_scene,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),
            resize_height=config.image_size,
            resize_width=config.image_size,
            time_step=config.num_frames,
        )
        test_loader = data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=config.num_workers, pin_memory=True, drop_last=False,
        )

        np_label     = np.load(os.path.join(label_path, scene_name + '.npy'),
                               allow_pickle=True)
        video_losses = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=scene_name, leave=False):
                frame = batch['standard'].to(device)[:, 0].float()
                t_tokens, latent = teacher(frame)
                s_tokens = student(latent)
                loss = (
                    F.mse_loss(s_tokens[0], t_tokens[0]) +
                    F.mse_loss(s_tokens[1], t_tokens[1])
                )
                video_losses.append(loss.item())

        os.makedirs(config.save_path, exist_ok=True)
        np.save(os.path.join(config.save_path, f'{scene_name}.npy'),
                np.array(video_losses))

        n = min(len(np_label), len(video_losses))
        all_labels.extend(np_label[-n:])
        all_losses.extend(video_losses[:n])

    all_labels = np.array(all_labels)
    all_losses = np.array(all_losses)

    # ── Metrics ────────────────────────────────────────────────────────────
    frame_auc     = roc_auc_score(all_labels, all_losses)
    avg_precision = average_precision_score(all_labels, all_losses)

    fpr, tpr, thresholds = roc_curve(all_labels, all_losses)
    fnr = 1 - tpr
    idx_eer   = np.nanargmin(np.abs(fnr - fpr))
    eer       = float(fpr[idx_eer])

    # Best-F1 threshold
    f1_scores = np.array([
        f1_score(all_labels, (all_losses >= t).astype(int), zero_division=0)
        for t in thresholds
    ])
    idx_f1       = np.argmax(f1_scores)
    best_f1      = float(f1_scores[idx_f1])
    best_thresh  = float(thresholds[idx_f1])

    print(f'\n  Variant={variant}  AUC={frame_auc:.4f}  PR-AUC={avg_precision:.4f}'
          f'  EER={eer:.4f}  Best-F1={best_f1:.4f} @ thr={best_thresh:.4f}')

    # ── Score plot ─────────────────────────────────────────────────────────
    norm_losses = (all_losses - all_losses.min()) / max(
        all_losses.max() - all_losses.min(), 1e-8
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(norm_losses,  label='Predicted scores', color='#1f77b4', lw=1.2)
    ax.plot(all_labels,   label='Ground truth',     color='#ff7f0e', lw=1.2)
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Normalised score / label')
    ax.set_title(f'LightViT-AD anomaly scores — {variant} '
                 f'(AUC={frame_auc:.3f}, PR-AUC={avg_precision:.3f})')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(config.save_path, f'score_plot_{variant}.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {plot_path}')

    return {
        'auc':            round(frame_auc,     4),
        'eer':            round(eer,            4),
        'avg_precision':  round(avg_precision,  4),
        'best_f1':        round(best_f1,        4),
        'best_threshold': round(best_thresh,    6),
    }


# ---------------------------------------------------------------------------
# Checkpoint I/O  (FIX-A: consistent keys for both train.py and deploy_jetson.py)
# ---------------------------------------------------------------------------

def save_checkpoint(
    teacher: ViTTeacher,
    student: ViTStudent,
    mean_std: dict,
    best_auc: float,
    path: str,
) -> None:
    """
    Save checkpoint with keys compatible with deploy_jetson.py loader.

    Keys: teacher_state_dict, student_state_dict, mean_std, best_auc
    (The original script saved only {'state_dict': student.state_dict()}
    which caused KeyError when loaded by deploy_jetson.py.)
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save({
        'teacher_state_dict': teacher.state_dict(),
        'student_state_dict': student.state_dict(),
        'mean_std':           mean_std,
        'best_auc':           best_auc,
    }, path)


def load_checkpoint(path: str, device: torch.device):
    """
    Load checkpoint saved by save_checkpoint().

    Returns: teacher_sd, student_sd, mean_std, best_auc
    """
    ckpt       = torch.load(path, map_location=device)
    teacher_sd = ckpt.get('teacher_state_dict')
    student_sd = ckpt.get('student_state_dict')
    mean_std   = ckpt.get('mean_std', {})
    best_auc   = ckpt.get('best_auc', 0.0)

    if teacher_sd is None or student_sd is None:
        raise KeyError(
            f"Checkpoint at '{path}' is missing 'teacher_state_dict' or "
            f"'student_state_dict'. Keys found: {list(ckpt.keys())}. "
            f"If this checkpoint was saved by the original training script "
            f"(key='state_dict'), please re-train with the fixed train_x86.py."
        )
    return teacher_sd, student_sd, mean_std, best_auc
