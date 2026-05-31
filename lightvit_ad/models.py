"""
lightvit_ad/models.py
=====================
ViTTeacher, ViTStudent, CombinedModel.

Fixes over the original train_val_VIT_Distillation_Autoencoder_base_tiny_optimized_trial.py
---------------------------------------------------------------------------
FIX-1  ViTStudent: block-slicing instead of timm create_model(depth=6) kwarg.
       The depth kwarg is not accepted on timm ≤0.3.x (JetPack 4.6) and raises
       TypeError: got multiple values for keyword argument 'embed_dim'.
       The slice approach works on all timm versions.

FIX-2  Token normalisation: MSE loss is intentionally computed on raw,
       unnormalised token vectors (no ℓ₂ normalisation before loss).
       This design choice is now documented explicitly (see revised manuscript
       Section 3.3) and is preserved here unchanged.

FIX-3  All system-size references use the combined teacher+student figure.
       The student alone (depth=6, embed_dim=192) ≈ 5.7 M parameters;
       the combined system ≈ 11.5 M parameters, 3.54 GFLOPs.
"""

import torch
import torch.nn as nn
import timm
from torch.nn import init


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module, vit_std: float = 0.02) -> None:
    """Kaiming/Xavier init for Conv, Linear, Norm and Embedding layers."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(module.weight, a=0, mode='fan_in',
                             nonlinearity='leaky_relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, (nn.InstanceNorm2d, nn.BatchNorm2d, nn.GroupNorm)):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        init.xavier_normal_(module.weight)
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=vit_std)
        if hasattr(module, 'padding_idx') and module.padding_idx is not None:
            init.constant_(module.weight[module.padding_idx], 0)
    elif isinstance(module, nn.LayerNorm):
        init.constant_(module.bias, 0)
        init.constant_(module.weight, 1)


# ---------------------------------------------------------------------------
# ViTTeacher
# ---------------------------------------------------------------------------

class ViTTeacher(nn.Module):
    """
    Pretrained, frozen DeiT-tiny teacher.

    Processes an input image through all 12 transformer blocks and returns:
      - (cls_token_out, dist_token_out) : raw final-layer tokens (B, 192)
      - latent_token                    : fused 192-dim representation (B, 192)

    The latent_token is the regression target passed to the student.
    The raw tokens are used to compute the MSE anomaly score at inference.
    """

    def __init__(
        self,
        model_name: str = 'deit_tiny_distilled_patch16_224',
        pretrained: bool = True,
        latent_dim: int = 192,
    ) -> None:
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.vit.head = nn.Identity()

        # Fusion MLP: concatenate CLS+DIST (384-dim) → 192-dim latent
        self.token_fusion = nn.Sequential(
            nn.Linear(self.vit.embed_dim * 2, latent_dim),
            nn.GELU(),
        )

        if not pretrained:
            self.vit.apply(lambda m: init_weights(m, vit_std=0.02))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, 224, 224) normalised input image

        Returns:
            (cls_token_out, dist_token_out): tuple of (B, D) tensors
            latent_token: (B, D) fused representation
        """
        x = self.vit.patch_embed(x)               # (B, N, D)
        B, N, D = x.shape
        cls_token  = self.vit.cls_token.expand(B, -1, -1)   # (B, 1, D)
        dist_token = self.vit.dist_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_token, dist_token, x), dim=1)    # (B, N+2, D)
        x = x + self.vit.pos_embed

        for block in self.vit.blocks:
            x = block(x)

        cls_token_out  = x[:, 0]   # (B, D)
        dist_token_out = x[:, 1]   # (B, D)

        # Fuse into a single latent token for the student
        latent_token = self.token_fusion(
            torch.cat((cls_token_out, dist_token_out), dim=-1)
        )  # (B, latent_dim)

        return (cls_token_out, dist_token_out), latent_token


# ---------------------------------------------------------------------------
# ViTStudent  (FIX-1: block slicing, not timm kwarg)
# ---------------------------------------------------------------------------

class ViTStudent(nn.Module):
    """
    Depth-reduced DeiT-tiny student (6 of 12 blocks).

    The student does NOT process raw image pixels at any point.
    Its sole input is the 192-dim latent token from the teacher,
    broadcast uniformly to all 196 patch positions, prepended with
    its own learnable CLS and DIST tokens.

    Input sequence:  [cls, dist, latent, latent, ...(×196)]
                      ∈ R^(B × 198 × 192)

    FIX-1: block slicing avoids the timm version-dependent depth= kwarg crash.
           timm.create_model is called with default depth=12, then blocks[:6]
           are retained.  The resulting student architecture is identical to
           using depth=6 on timm ≥0.6.x but works on all timm versions.
    """

    def __init__(
        self,
        model_name: str = 'deit_tiny_distilled_patch16_224',
        pretrained: bool = False,
        depth: int = 6,
    ) -> None:
        super().__init__()
        # Create full 12-block model, then slice to `depth` blocks (FIX-1)
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.vit.head = nn.Identity()
        self.vit.blocks = self.vit.blocks[:depth]   # ← FIX-1: safe on all timm versions

        if not pretrained:
            self.vit.apply(lambda m: init_weights(m, vit_std=0.02))

    def forward(self, latent_feature: torch.Tensor):
        """
        Args:
            latent_feature: (B, 192) latent token from teacher

        Returns:
            (cls_token_out, dist_token_out): tuple of (B, 192) tensors
        """
        B = latent_feature.shape[0]
        cls_token  = self.vit.cls_token.expand(B, -1, -1)   # (B, 1, 192)
        dist_token = self.vit.dist_token.expand(B, -1, -1)  # (B, 1, 192)

        # Broadcast latent to all 196 patch positions
        latent_expanded = latent_feature.unsqueeze(1).expand(-1, 196, -1)  # (B, 196, 192)

        x = torch.cat([cls_token, dist_token, latent_expanded], dim=1)    # (B, 198, 192)
        x = x + self.vit.pos_embed

        for block in self.vit.blocks:
            x = block(x)

        cls_token_out  = x[:, 0]   # (B, 192)
        dist_token_out = x[:, 1]   # (B, 192)

        return (cls_token_out, dist_token_out)


# ---------------------------------------------------------------------------
# CombinedModel  (used for TorchScript tracing and benchmarking)
# ---------------------------------------------------------------------------

class CombinedModel(nn.Module):
    """
    Wraps teacher + student into a single nn.Module for TorchScript tracing
    and unified latency/throughput benchmarking.

    forward(x) performs:
      1. Teacher forward pass (image → latent token + raw CLS/DIST tokens)
      2. Student forward pass (latent token → reproduced CLS/DIST tokens)
      Returns: (s_cls, s_dist) — student output tokens
    """

    def __init__(self, teacher: ViTTeacher, student: ViTStudent) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student

    def forward(self, x: torch.Tensor):
        t_tokens, latent = self.teacher(x)
        s_tokens = self.student(latent)
        return s_tokens


# ---------------------------------------------------------------------------
# Anomaly score
# ---------------------------------------------------------------------------

def anomaly_score(
    teacher_tokens: tuple,
    student_tokens: tuple,
) -> torch.Tensor:
    """
    Per-frame anomaly score: MSE between teacher and student CLS/DIST tokens.

    Score is computed on raw, unnormalised token vectors (no ℓ₂ normalisation).
    This makes the score sensitive to both the direction and magnitude of
    deviation from the teacher manifold.

    Args:
        teacher_tokens: (t_cls, t_dist) each (B, 192)
        student_tokens: (s_cls, s_dist) each (B, 192)

    Returns:
        score: (B,) per-sample anomaly score
    """
    import torch.nn.functional as F
    t_cls,  t_dist  = teacher_tokens
    s_cls,  s_dist  = student_tokens
    score = (
        F.mse_loss(s_cls,  t_cls,  reduction='none').mean(dim=-1) +
        F.mse_loss(s_dist, t_dist, reduction='none').mean(dim=-1)
    )
    return score   # (B,)
