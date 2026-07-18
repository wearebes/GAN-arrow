"""Pure-PyTorch BGC augmentation and ADA probability controller.

The transform families and default distributions follow the ``bgc`` pipeline
from NVIDIA's StyleGAN2-ADA reference implementation:
https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/augment.py

The adaptive update follows the paper/reference defaults (target 0.6, update
interval 4, 500 kimg adjustment speed).  The geometric resampling uses standard
PyTorch ``grid_sample`` with reflection padding instead of NVIDIA's custom
upfirdn2d CUDA kernels, so this module remains portable to CPU/MPS/CUDA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F


def _identity(batch_size: int, size: int, *, device, dtype):
    matrix = torch.eye(size, device=device, dtype=dtype)
    return matrix.unsqueeze(0).repeat(batch_size, 1, 1)


def _compose(current, transform):
    return torch.bmm(current, transform)


def _translation(tx, ty, *, device, dtype):
    batch_size = tx.shape[0]
    matrix = _identity(batch_size, 3, device=device, dtype=dtype)
    matrix[:, 0, 2] = tx
    matrix[:, 1, 2] = ty
    return matrix


def _scale(sx, sy, *, device, dtype):
    batch_size = sx.shape[0]
    matrix = _identity(batch_size, 3, device=device, dtype=dtype)
    matrix[:, 0, 0] = sx
    matrix[:, 1, 1] = sy
    return matrix


def _rotation(theta, *, device, dtype):
    batch_size = theta.shape[0]
    matrix = _identity(batch_size, 3, device=device, dtype=dtype)
    cosine = torch.cos(theta)
    sine = torch.sin(theta)
    matrix[:, 0, 0] = cosine
    matrix[:, 0, 1] = -sine
    matrix[:, 1, 0] = sine
    matrix[:, 1, 1] = cosine
    return matrix


def _active(batch_size: int, probability: float, *, device):
    if probability <= 0:
        return torch.zeros(batch_size, device=device, dtype=torch.bool)
    if probability >= 1:
        return torch.ones(batch_size, device=device, dtype=torch.bool)
    return torch.rand(batch_size, device=device) < probability


class AdaBcgAugment:
    """Differentiable pixel-blit, geometry, and color transforms."""

    def __call__(self, images: torch.Tensor, probability: float):
        probability = float(max(0.0, min(1.0, probability)))
        if probability <= 0:
            return images
        images = self._geometry(images, probability)
        return self._color(images, probability)

    def _geometry(self, images: torch.Tensor, probability: float):
        batch_size, _, height, width = images.shape
        device = images.device
        dtype = images.dtype
        matrix = _identity(batch_size, 3, device=device, dtype=dtype)

        # Pixel blitting: x-flip, 90-degree rotation, integer translation.
        active = _active(batch_size, probability, device=device)
        flip = torch.where(active, -torch.ones(batch_size, device=device, dtype=dtype), torch.ones(batch_size, device=device, dtype=dtype))
        matrix = _compose(matrix, _scale(flip, torch.ones_like(flip), device=device, dtype=dtype))

        active = _active(batch_size, probability, device=device)
        quarter_turns = torch.randint(0, 4, (batch_size,), device=device)
        theta = torch.where(active, quarter_turns.to(dtype) * (math.pi / 2), torch.zeros(batch_size, device=device, dtype=dtype))
        matrix = _compose(matrix, _rotation(theta, device=device, dtype=dtype))

        active = _active(batch_size, probability, device=device)
        tx_pixels = torch.round((torch.rand(batch_size, device=device, dtype=dtype) * 2 - 1) * 0.125 * width)
        ty_pixels = torch.round((torch.rand(batch_size, device=device, dtype=dtype) * 2 - 1) * 0.125 * height)
        tx = torch.where(active, 2 * tx_pixels / max(width, 1), torch.zeros_like(tx_pixels))
        ty = torch.where(active, 2 * ty_pixels / max(height, 1), torch.zeros_like(ty_pixels))
        matrix = _compose(matrix, _translation(tx, ty, device=device, dtype=dtype))

        # General geometry: isotropic scale, rotation, anisotropy, fractional translation.
        active = _active(batch_size, probability, device=device)
        sampled = torch.exp2(torch.randn(batch_size, device=device, dtype=dtype) * 0.2)
        sampled = torch.where(active, 1 / sampled, torch.ones_like(sampled))
        matrix = _compose(matrix, _scale(sampled, sampled, device=device, dtype=dtype))

        rotation_probability = 1 - math.sqrt(max(0.0, 1 - probability))
        active = _active(batch_size, rotation_probability, device=device)
        theta = (torch.rand(batch_size, device=device, dtype=dtype) * 2 - 1) * math.pi
        theta = torch.where(active, theta, torch.zeros_like(theta))
        matrix = _compose(matrix, _rotation(theta, device=device, dtype=dtype))

        active = _active(batch_size, probability, device=device)
        sampled = torch.exp2(torch.randn(batch_size, device=device, dtype=dtype) * 0.2)
        sx = torch.where(active, 1 / sampled, torch.ones_like(sampled))
        sy = torch.where(active, sampled, torch.ones_like(sampled))
        matrix = _compose(matrix, _scale(sx, sy, device=device, dtype=dtype))

        active = _active(batch_size, rotation_probability, device=device)
        theta = (torch.rand(batch_size, device=device, dtype=dtype) * 2 - 1) * math.pi
        theta = torch.where(active, theta, torch.zeros_like(theta))
        matrix = _compose(matrix, _rotation(theta, device=device, dtype=dtype))

        active = _active(batch_size, probability, device=device)
        tx = torch.where(active, torch.randn(batch_size, device=device, dtype=dtype) * 0.125 * 2, torch.zeros(batch_size, device=device, dtype=dtype))
        ty = torch.where(active, torch.randn(batch_size, device=device, dtype=dtype) * 0.125 * 2, torch.zeros(batch_size, device=device, dtype=dtype))
        matrix = _compose(matrix, _translation(tx, ty, device=device, dtype=dtype))

        grid = F.affine_grid(matrix[:, :2, :], images.shape, align_corners=False)
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    def _color(self, images: torch.Tensor, probability: float):
        batch_size, channels, _, _ = images.shape
        device = images.device
        dtype = images.dtype

        active = _active(batch_size, probability, device=device).view(-1, 1, 1, 1)
        brightness = torch.randn(batch_size, 1, 1, 1, device=device, dtype=dtype) * 0.2
        images = images + torch.where(active, brightness, torch.zeros_like(brightness))

        active = _active(batch_size, probability, device=device).view(-1, 1, 1, 1)
        contrast = torch.exp2(torch.randn(batch_size, 1, 1, 1, device=device, dtype=dtype) * 0.5)
        images = images * torch.where(active, contrast, torch.ones_like(contrast))
        if channels < 3:
            return images

        # RGB color transforms use the luma axis from the NVIDIA reference.
        rgb = images[:, :3]
        axis = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=dtype)
        axis = axis / torch.linalg.vector_norm(axis)
        projection = torch.outer(axis, axis)
        identity = torch.eye(3, device=device, dtype=dtype)

        active = _active(batch_size, probability, device=device)
        flip_matrix = identity - 2 * projection
        color_matrix = torch.where(active.view(-1, 1, 1), flip_matrix, identity).clone()

        active = _active(batch_size, probability, device=device)
        theta = (torch.rand(batch_size, device=device, dtype=dtype) * 2 - 1) * math.pi
        theta = torch.where(active, theta, torch.zeros_like(theta))
        kx, ky, kz = axis
        skew = torch.tensor([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]], device=device, dtype=dtype)
        hue = (
            identity.unsqueeze(0) * torch.cos(theta).view(-1, 1, 1)
            + (1 - torch.cos(theta)).view(-1, 1, 1) * projection.unsqueeze(0)
            + torch.sin(theta).view(-1, 1, 1) * skew.unsqueeze(0)
        )
        color_matrix = torch.bmm(hue, color_matrix)

        active = _active(batch_size, probability, device=device)
        saturation = torch.exp2(torch.randn(batch_size, device=device, dtype=dtype))
        saturation = torch.where(active, saturation, torch.ones_like(saturation))
        sat_matrix = projection.unsqueeze(0) + (identity - projection).unsqueeze(0) * saturation.view(-1, 1, 1)
        color_matrix = torch.bmm(sat_matrix, color_matrix)

        flat = rgb.reshape(batch_size, 3, -1)
        rgb = torch.bmm(color_matrix, flat).reshape_as(rgb)
        if channels == 3:
            return rgb
        return torch.cat([rgb, images[:, 3:]], dim=1)


@dataclass
class AdaController:
    """Adaptive probability update using the StyleGAN2-ADA r_t heuristic."""

    target: float = 0.6
    interval: int = 4
    speed_kimg: float = 500.0
    probability: float = 0.0
    pending_sign_sum: float = 0.0
    pending_count: int = 0
    pending_batches: int = 0
    last_rt: float | None = None

    def __post_init__(self):
        if not 0 <= self.target <= 1:
            raise ValueError("ADA target must be in [0, 1]")
        if self.interval < 1:
            raise ValueError("ADA interval must be at least 1")
        if self.speed_kimg <= 0:
            raise ValueError("ADA speed_kimg must be positive")
        if not 0 <= self.probability <= 1:
            raise ValueError("ADA initial probability must be in [0, 1]")

    def observe(self, real_logits: torch.Tensor):
        signs = torch.sign(real_logits.detach())
        self.pending_sign_sum += float(signs.sum().item())
        self.pending_count += int(signs.numel())
        self.pending_batches += 1
        if self.pending_batches < self.interval:
            return None

        rt = self.pending_sign_sum / max(self.pending_count, 1)
        direction = 1.0 if rt > self.target else -1.0 if rt < self.target else 0.0
        adjustment = direction * self.pending_count / (self.speed_kimg * 1000.0)
        self.probability = float(max(0.0, min(1.0, self.probability + adjustment)))
        self.last_rt = float(rt)
        result = {"rt": self.last_rt, "p": self.probability, "observed_images": self.pending_count}
        self.pending_sign_sum = 0.0
        self.pending_count = 0
        self.pending_batches = 0
        return result

    def state_dict(self):
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)
