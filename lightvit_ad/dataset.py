"""
lightvit_ad/dataset.py
======================
Self-contained DataLoader and mean/std computation.
No external project modules (config, data_utils_norm_n, utils) required.
"""

import os
import glob
import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Mean / Std computation (from training normal frames)
# ---------------------------------------------------------------------------

def compute_mean_and_std(
    train_folder: str,
    resize_height: int,
    resize_width: int,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Compute per-channel mean and std from a folder of normal training frames.

    The folder must be organised as:
        train_folder/
            <any_subfolder>/
                frame_001.jpg
                frame_002.jpg
                ...
    (ImageFolder-compatible: one level of subdirectories.)

    Returns:
        mean: (3,) CPU tensor
        std:  (3,) CPU tensor
    """
    transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(train_folder, transform=transform)
    loader  = data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    channels     = dataset[0][0].shape[0]
    mean         = torch.zeros(channels).to(device)
    std          = torch.zeros(channels).to(device)
    total_pixels = 0

    with torch.no_grad():
        for images, _ in tqdm(loader, desc='Computing mean/std'):
            images = images.to(device)
            B, C, H, W  = images.shape
            total_pixels += B * H * W
            mean.add_(images.sum(dim=[0, 2, 3]))
            std.add_((images ** 2).sum(dim=[0, 2, 3]))

    mean /= total_pixels
    std   = torch.sqrt(std / total_pixels - mean ** 2)
    return mean.cpu(), std.cpu()


# ---------------------------------------------------------------------------
# Frame loading
# ---------------------------------------------------------------------------

def np_load_frame(
    filename: str,
    resize_height: int,
    resize_width: int,
):
    """
    Load a BGR image (cv2), convert to RGB, resize, return as PIL Image.
    """
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f'Image not found: {filename}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize_width, resize_height))
    return Image.fromarray(img.astype(np.uint8))


# ---------------------------------------------------------------------------
# DataLoader  (frame-based, supports return_image_path)
# ---------------------------------------------------------------------------

class DataLoader(data.Dataset):
    """
    Frame-level dataset for anomaly detection.

    Expects either:
      video_folder/                    # flat folder of frames
          0001.jpg, 0002.jpg, ...
    or:
      video_folder/
          scene_01/
              0001.jpg, ...
          scene_02/
              ...

    Returns dict with key 'standard': (T, 3, H, W) tensor.
    If return_image_path=True, also returns 'image_path': str.
    """

    def __init__(
        self,
        video_folder: str,
        transform,
        resize_height: int,
        resize_width: int,
        time_step: int = 1,
        num_pred: int = 0,
        return_image_path: bool = False,
    ) -> None:
        self.dir               = video_folder
        self.transform         = transform
        self._resize_height    = resize_height
        self._resize_width     = resize_width
        self._time_step        = time_step
        self._num_pred         = num_pred
        self.return_image_path = return_image_path
        self.video_frames: list = []
        self._setup()

    def _setup(self) -> None:
        candidates = sorted(glob.glob(os.path.join(self.dir, '*')))
        frames     = []

        if candidates and os.path.isdir(candidates[0]):
            for d in candidates:
                sub = sorted(
                    glob.glob(os.path.join(d, '*.jpg')) +
                    glob.glob(os.path.join(d, '*.png')),
                    key=lambda p: int(
                        os.path.splitext(os.path.basename(p))[0].split('_')[-1]
                    )
                )
                frames.extend(sub)
        else:
            frames = sorted(
                candidates,
                key=lambda p: int(
                    os.path.splitext(os.path.basename(p))[0].split('_')[-1]
                )
            )

        self.video_frames = frames
        n = len(frames)
        need = self._time_step + self._num_pred
        if n < need:
            raise RuntimeError(
                f'Not enough frames ({n}) for time_step={self._time_step} '
                f'+ num_pred={self._num_pred} in {self.dir}'
            )

    def __len__(self) -> int:
        return max(len(self.video_frames) - self._time_step - self._num_pred + 1, 0)

    def __getitem__(self, index: int):
        batch: list = []
        for i in range(self._time_step + self._num_pred):
            img = np_load_frame(
                self.video_frames[index + i],
                self._resize_height,
                self._resize_width,
            )
            if self.transform:
                img = self.transform(img)
            batch.append(img)

        result = {'standard': torch.stack(batch, dim=0)}

        if self.return_image_path:
            result['image_path'] = self.video_frames[index]

        return result
