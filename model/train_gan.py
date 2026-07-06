import argparse
import copy
import csv
import json
import math
import os
import random
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROCESSED_IMAGE_EXTENSION = ".png"
PREPARED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
PREPROCESS_VERSION = "target_roi_v3_lossless_png"


def count_images(path: Path) -> int:
    return sum(1 for item in Path(path).iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def prepared_image_paths(image_dir: Path):
    preferred_by_stem = {}
    extension_rank = {extension: rank for rank, extension in enumerate(PREPARED_IMAGE_EXTENSIONS)}
    for path in Path(image_dir).iterdir():
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in extension_rank:
            continue
        current = preferred_by_stem.get(path.stem)
        if current is None or extension_rank[suffix] < extension_rank[current.suffix.lower()]:
            preferred_by_stem[path.stem] = path
    return [preferred_by_stem[stem] for stem in sorted(preferred_by_stem)]


def _remove_prepared_cache_for_stem(processed_dir: Path, stem: str, keep: Path | None = None):
    for extension in PREPARED_IMAGE_EXTENSIONS:
        candidate = processed_dir / f"{stem}{extension}"
        if keep is not None and candidate == keep:
            continue
        candidate.unlink(missing_ok=True)


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: Path = Path("dataset/origin_data")
    processed_dir: Path = Path("dataset/generate_data/processed_256")
    output_dir: Path = Path("outputs")
    image_size: int = 256
    latent_dim: int = 100
    channels: int = 3
    batch_size: int = 16
    epochs: int = 18
    d_lr: float = 0.0001
    g_lr: float = 0.0002
    beta1: float = 0.5
    real_label: float = 0.9
    fake_label: float = 0.0
    discriminator_norm: str = "none"
    discriminator_mode: str = "global"
    adversarial_loss_mode: str = "bce"
    generator_mode: str = "upsample"
    generator_features: int = 32
    discriminator_features: int = 32
    diffaugment: bool = False
    diffaugment_policy: str = "color,translation,cutout"
    ema_decay: float = 0.0
    target_prior_weight: float = 0.0
    amp: bool = False
    grad_accum_steps: int = 1
    sample_interval: int = 1
    checkpoint_interval: int = 1
    freeze_discriminator_during_generator_step: bool = True
    skip_prepare: bool = False
    max_steps: int | None = None
    resume_generator: Path | None = None
    resume_discriminator: Path | None = None
    resume_ema_generator: Path | None = None
    resume_training_state: Path | None = None
    min_target_anchor_fraction: float = 0.01
    target_crop_expansion: float = 2.9
    seed: int = 42
    workers: int = 0

    def __post_init__(self):
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be at least 1")
        if self.sample_interval < 1:
            raise ValueError("sample_interval must be at least 1")
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be at least 1")

    @classmethod
    def from_dataset_size(cls, dataset_size: int, **overrides):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")

        defaults = {}
        if dataset_size < 300:
            defaults.update(
                batch_size=16,
                epochs=18,
                image_size=256,
                processed_dir=Path("dataset/generate_data/processed_256"),
                discriminator_norm="none",
                d_lr=0.0001,
                g_lr=0.0002,
            )
        else:
            defaults.update(
                batch_size=32,
                epochs=12,
                image_size=256,
                processed_dir=Path("dataset/generate_data/processed_256"),
                discriminator_norm="batch",
                d_lr=0.0002,
                g_lr=0.0002,
            )
        defaults.update(overrides)
        return cls(**defaults)


def _norm_layer(norm: str, channels: int):
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {norm}")


def _validate_power_of_two_image_size(image_size: int):
    if image_size < 64 or image_size & (image_size - 1) != 0:
        raise ValueError("image_size must be a power of two and at least 64")


class GeneratorProjectBlock(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, out_channels * 4 * 4, bias=False),
            nn.Unflatten(1, (out_channels, 4, 4)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, noise):
        return self.layers(noise)


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class TransposeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class DownsampleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm="none"):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
            _norm_layer(norm, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, image_size=256, latent_dim=100, channels=3, features=32, mode="upsample"):
        _validate_power_of_two_image_size(image_size)
        super().__init__()
        self.output_size = image_size
        self.latent_dim = latent_dim
        self.mode = mode
        self.blocks = nn.ModuleList()

        if mode == "upsample":
            self.project = GeneratorProjectBlock(latent_dim, features * 8)
            current_channels = features * 8
            current_size = 4
            while current_size < image_size:
                next_channels = max(features, current_channels // 2)
                self.blocks.append(UpsampleConvBlock(current_channels, next_channels))
                current_channels = next_channels
                current_size *= 2
            self.to_rgb = nn.Sequential(nn.Conv2d(current_channels, channels, 3, 1, 1, bias=False), nn.Tanh())
        elif mode == "transpose":
            self.project = TransposeConvBlock(latent_dim, features * 8, kernel_size=4, stride=1, padding=0)
            current_channels = features * 8
            current_size = 4
            while current_size < image_size // 2:
                next_channels = max(features, current_channels // 2)
                self.blocks.append(TransposeConvBlock(current_channels, next_channels))
                current_channels = next_channels
                current_size *= 2
            self.to_rgb = nn.Sequential(nn.ConvTranspose2d(current_channels, channels, 4, 2, 1, bias=False), nn.Tanh())
        else:
            raise ValueError(f"Unsupported generator mode: {mode}")

    def forward(self, x):
        if self.mode == "upsample" and x.dim() == 4 and x.size(-1) == 1 and x.size(-2) == 1:
            x = x.view(x.size(0), x.size(1))
        if self.mode == "transpose" and x.dim() == 2:
            x = x.view(x.size(0), x.size(1), 1, 1)
        x = self.project(x)
        for block in self.blocks:
            x = block(x)
        return self.to_rgb(x)


class Discriminator(nn.Module):
    def __init__(self, image_size=256, channels=3, features=32, norm="none"):
        _validate_power_of_two_image_size(image_size)
        super().__init__()
        self.input_size = image_size
        self.from_rgb = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.ModuleList()
        current_channels = features
        current_size = image_size // 2
        while current_size > 4:
            next_channels = min(features * 8, current_channels * 2)
            self.blocks.append(DownsampleConvBlock(current_channels, next_channels, norm=norm))
            current_channels = next_channels
            current_size //= 2
        self.classifier = nn.Conv2d(current_channels, 1, 4, 1, 0, bias=False)

    def forward(self, x):
        x = self.from_rgb(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x).view(-1)


class PatchDiscriminator(nn.Module):
    def __init__(self, image_size=256, channels=3, features=32, norm="none"):
        _validate_power_of_two_image_size(image_size)
        super().__init__()
        self.input_size = image_size
        self.from_rgb = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.ModuleList()
        current_channels = features
        current_size = image_size // 2
        while current_size > 16:
            next_channels = min(features * 8, current_channels * 2)
            self.blocks.append(DownsampleConvBlock(current_channels, next_channels, norm=norm))
            current_channels = next_channels
            current_size //= 2
        self.patch_head = nn.Conv2d(current_channels, 1, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.from_rgb(x)
        for block in self.blocks:
            x = block(x)
        return self.patch_head(x).flatten(1)


class ArrowImageDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int):
        self.paths = prepared_image_paths(image_dir)
        if not self.paths:
            raise ValueError(f"No prepared images found in {image_dir}")
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def image_has_signal(path: Path, min_std=1.0, min_max=16) -> bool:
    with Image.open(path) as image:
        stat = transforms.functional.pil_to_tensor(image.convert("RGB")).float()
    return bool(stat.max().item() >= min_max and stat.std().item() >= min_std)


def _probe_heic_video_streams(source_path: Path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v",
        "-show_entries",
        "stream=index,width,height",
        "-of",
        "csv=p=0",
        str(source_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    streams = []
    for line in result.stdout.splitlines():
        match = re.match(r"^(\d+),(\d+),(\d+)", line.strip())
        if match:
            streams.append(tuple(int(part) for part in match.groups()))
    return streams


def _decode_heic_stream(source_path: Path, output_path: Path, stream_index: int | None):
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-threads", "1", "-i", str(source_path)]
    if stream_index is not None:
        command.extend(["-map", f"0:{stream_index}"])
    command.extend(["-frames:v", "1", str(output_path)])
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _candidate_heic_stream_indices(streams):
    if not streams:
        return [14, 15, 24, 45]

    available = {index for index, _, _ in streams}
    preferred = [14, 15, 24, 45]
    return [index for index in preferred if index in available]


def _score_decoded_candidate(path: Path):
    if not path.exists() or not image_has_signal(path):
        return -1.0
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
        anchor_pixels = int(_target_roi_anchor_mask(array).sum())
        blue_pixels = int((_target_color_component_masks(array)[2] & _target_roi_region(array)).sum())
        if anchor_pixels < 100:
            return -1.0
        area = image.width * image.height
    return float(anchor_pixels + blue_pixels * 3 + area * 1e-5)


def _convert_heic_with_ffmpeg(source_path: Path, output_path: Path):
    default_temp = output_path.with_name(f"{output_path.stem}.stream_default{PROCESSED_IMAGE_EXTENSION}")
    try:
        _decode_heic_stream(source_path, default_temp, None)
        if image_has_signal(default_temp):
            default_temp.replace(output_path)
            return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    default_temp.unlink(missing_ok=True)

    streams = _probe_heic_video_streams(source_path)
    stream_indices = _candidate_heic_stream_indices(streams)

    last_error = None
    best_temp = None
    best_score = -1.0
    temp_paths = []
    for stream_index in stream_indices:
        temp_path = output_path.with_name(
            f"{output_path.stem}.stream_{stream_index if stream_index is not None else 'default'}"
            f"{PROCESSED_IMAGE_EXTENSION}"
        )
        temp_paths.append(temp_path)
        try:
            _decode_heic_stream(source_path, temp_path, stream_index)
            score = _score_decoded_candidate(temp_path)
            if score > best_score:
                best_score = score
                best_temp = temp_path
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            last_error = exc
    if best_temp is not None and best_score > 0:
        best_temp.replace(output_path)
        for temp_path in temp_paths:
            if temp_path != output_path:
                temp_path.unlink(missing_ok=True)
        return
    for temp_path in temp_paths:
        temp_path.unlink(missing_ok=True)
    raise RuntimeError(f"ffmpeg could not decode a non-black image from {source_path}: {last_error}")


def _convert_with_pillow(source_path: Path, output_path: Path):
    with Image.open(source_path) as image:
        image.convert("RGB").save(output_path, format="PNG")


def _target_hsv_components(array):
    hsv = np.asarray(Image.fromarray(array.astype(np.uint8)).convert("HSV"))
    hue = hsv[:, :, 0].astype(np.int16)
    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)
    return hue, saturation, value


def _target_anchor_mask(array):
    red_ring, yellow_ring, _ = _target_color_component_masks(array)
    return red_ring | yellow_ring


def _target_color_component_masks(array):
    hue, saturation, value = _target_hsv_components(array)
    strong_color = (saturation > 85) & (value > 90)
    red_ring = ((hue < 12) | (hue > 242)) & strong_color
    yellow_ring = (hue >= 24) & (hue <= 48) & strong_color
    blue_ring = (hue >= 125) & (hue <= 165) & strong_color
    return red_ring, yellow_ring, blue_ring


def _target_roi_anchor_mask(array):
    height, width = array.shape[:2]
    return _target_anchor_mask(array) & _target_roi_region(array)


def _target_roi_region(array):
    height, width = array.shape[:2]
    yy, xx = np.indices((height, width))
    logo_region = (xx > width * 0.72) & (yy < height * 0.22)
    central_region = (
        (xx > width * 0.08)
        & (xx < width * 0.92)
        & (yy > height * 0.12)
        & (yy < height * 0.88)
    )
    return central_region & ~logo_region


def _target_color_mask(array):
    red_ring, yellow_ring, blue_ring = _target_color_component_masks(array)
    return red_ring | yellow_ring | blue_ring


def image_has_target_anchor(path: Path, min_pixels=100, min_fraction=0.01) -> bool:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    red_ring, yellow_ring, blue_ring = _target_color_component_masks(array)
    anchor = red_ring | yellow_ring
    anchor_pixels = int(anchor.sum())
    blue_pixels = int(blue_ring.sum())
    area = array.shape[0] * array.shape[1]
    blue_fraction = blue_pixels / area
    return bool(
        anchor_pixels >= min_pixels
        and anchor_pixels / area >= min_fraction
        and blue_pixels >= 50
        and blue_fraction >= 0.003
    )


def _crop_target_roi_file(path: Path, target_crop_expansion: float = 2.9):
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(image)

    mask = _target_roi_anchor_mask(array)
    y_indices, x_indices = np.where(mask)
    if len(x_indices) < 100:
        return False

    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    color_extent = max(x_max - x_min + 1, y_max - y_min + 1)
    crop_size = int(color_extent * target_crop_expansion)
    crop_size = max(crop_size, 180)
    crop_size = min(crop_size, image.width, image.height)

    left = int(round(center_x - crop_size / 2))
    top = int(round(center_y - crop_size / 2))
    left = max(0, min(left, image.width - crop_size))
    top = max(0, min(top, image.height - crop_size))
    cropped = image.crop((left, top, left + crop_size, top + crop_size))
    cropped.save(path, format="PNG")
    return True


def prepare_images(
    source_dir: Path,
    processed_dir: Path,
    image_size: int,
    limit: int | None = None,
    min_target_anchor_fraction: float = 0.01,
    target_crop_expansion: float = 2.9,
) -> int:
    processed_dir.mkdir(parents=True, exist_ok=True)
    version_path = processed_dir / ".preprocess_version"
    preprocess_version = (
        f"{PREPROCESS_VERSION}:min_anchor_fraction={min_target_anchor_fraction:.4f}:"
        f"crop_expansion={target_crop_expansion:.4f}"
    )
    cache_is_current = version_path.exists() and version_path.read_text().strip() == preprocess_version
    prepared_count = 0
    source_paths = sorted(
        path for path in Path(source_dir).iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not source_paths:
        raise ValueError(f"No source images found in {source_dir}")
    if limit is not None:
        source_paths = source_paths[:limit]

    for source_path in source_paths:
        output_path = processed_dir / f"{source_path.stem}{PROCESSED_IMAGE_EXTENSION}"
        if (
            cache_is_current
            and output_path.exists()
            and output_path.stat().st_mtime >= source_path.stat().st_mtime
            and image_has_signal(output_path)
            and image_has_target_anchor(output_path, min_fraction=min_target_anchor_fraction)
        ):
            _remove_prepared_cache_for_stem(processed_dir, source_path.stem, keep=output_path)
            prepared_count += 1
            continue
        try:
            if source_path.suffix.lower() in {".heic", ".heif"}:
                _convert_heic_with_ffmpeg(source_path, output_path)
            else:
                _convert_with_pillow(source_path, output_path)
        except RuntimeError:
            output_path.unlink(missing_ok=True)
            _remove_prepared_cache_for_stem(processed_dir, source_path.stem)
            continue
        if not _crop_target_roi_file(output_path, target_crop_expansion=target_crop_expansion):
            output_path.unlink(missing_ok=True)
            _remove_prepared_cache_for_stem(processed_dir, source_path.stem)
            continue
        if not image_has_target_anchor(output_path, min_fraction=min_target_anchor_fraction):
            output_path.unlink(missing_ok=True)
            _remove_prepared_cache_for_stem(processed_dir, source_path.stem)
            continue
        if not image_has_signal(output_path):
            raise RuntimeError(f"Prepared image is blank or invalid: {output_path}")
        _remove_prepared_cache_for_stem(processed_dir, source_path.stem, keep=output_path)
        prepared_count += 1
    version_path.write_text(preprocess_version)
    return prepared_count


def weights_init(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0)


def adversarial_loss(output, label: float, mode="bce"):
    targets = torch.full_like(output, label, dtype=torch.float)
    if mode == "bce":
        return nn.functional.binary_cross_entropy_with_logits(output, targets)
    if mode == "lsgan":
        return nn.functional.mse_loss(output, targets)
    raise ValueError(f"Unsupported adversarial loss mode: {mode}")


def discriminator_log_loss(loss_real, loss_fake):
    return (loss_real + loss_fake) * 0.5


def discriminator_confidence(output, mode="bce"):
    if mode == "bce":
        return torch.sigmoid(output).mean().item()
    if mode == "lsgan":
        return output.detach().clamp(0.0, 1.0).mean().item()
    raise ValueError(f"Unsupported adversarial loss mode: {mode}")


def _diffaugment_color(images):
    batch_size = images.size(0)
    brightness = (torch.rand(batch_size, 1, 1, 1, device=images.device) - 0.5) * 0.4
    images = images + brightness

    per_pixel_mean = images.mean(dim=1, keepdim=True)
    saturation = torch.rand(batch_size, 1, 1, 1, device=images.device) * 0.4 + 0.8
    images = (images - per_pixel_mean) * saturation + per_pixel_mean

    per_image_mean = images.mean(dim=(1, 2, 3), keepdim=True)
    contrast = torch.rand(batch_size, 1, 1, 1, device=images.device) * 0.4 + 0.8
    return (images - per_image_mean) * contrast + per_image_mean


def _diffaugment_translation(images, ratio=0.125):
    _, _, height, width = images.shape
    max_shift_y = max(1, int(height * ratio))
    max_shift_x = max(1, int(width * ratio))
    shifted_images = []
    for image in images:
        shift_y = int(torch.randint(-max_shift_y, max_shift_y + 1, (1,), device=images.device).item())
        shift_x = int(torch.randint(-max_shift_x, max_shift_x + 1, (1,), device=images.device).item())
        shifted = torch.roll(image, shifts=(shift_y, shift_x), dims=(1, 2))
        if shift_y > 0:
            shifted[:, :shift_y, :] = 0
        elif shift_y < 0:
            shifted[:, shift_y:, :] = 0
        if shift_x > 0:
            shifted[:, :, :shift_x] = 0
        elif shift_x < 0:
            shifted[:, :, shift_x:] = 0
        shifted_images.append(shifted)
    return torch.stack(shifted_images, dim=0)


def _diffaugment_cutout(images, ratio=0.25):
    batch_size, _, height, width = images.shape
    cutout_h = max(1, int(height * ratio))
    cutout_w = max(1, int(width * ratio))
    mask = torch.ones_like(images)
    for index in range(batch_size):
        center_y = int(torch.randint(0, height, (1,), device=images.device).item())
        center_x = int(torch.randint(0, width, (1,), device=images.device).item())
        top = max(0, center_y - cutout_h // 2)
        bottom = min(height, top + cutout_h)
        left = max(0, center_x - cutout_w // 2)
        right = min(width, left + cutout_w)
        mask[index, :, top:bottom, left:right] = 0
    return images * mask


def diff_augment(images, policy="color,translation,cutout"):
    if not policy or policy == "none":
        return images
    for item in [part.strip() for part in policy.split(",") if part.strip()]:
        if item == "color":
            images = _diffaugment_color(images)
        elif item == "translation":
            images = _diffaugment_translation(images)
        elif item == "cutout":
            images = _diffaugment_cutout(images)
        else:
            raise ValueError(f"Unsupported DiffAugment policy: {item}")
    return images


def target_ring_prior_loss(images):
    images_01 = (images + 1.0) * 0.5
    _, _, height, width = images.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=images.device, dtype=images.dtype),
        torch.arange(width, device=images.device, dtype=images.dtype),
        indexing="ij",
    )
    center_y = (height - 1) * 0.5
    center_x = (width - 1) * 0.5
    radius = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    scale = min(height, width) / 256.0

    yellow_mask = radius <= 22.0 * scale
    red_mask = (radius > 22.0 * scale) & (radius <= 46.0 * scale)
    blue_mask = (radius > 46.0 * scale) & (radius <= 70.0 * scale)
    background_mask = (radius > 70.0 * scale) & (radius <= 105.0 * scale)

    def masked_color_loss(mask, color):
        target_color = torch.tensor(color, device=images.device, dtype=images.dtype).view(1, 3, 1)
        pixels = images_01[:, :, mask]
        return nn.functional.l1_loss(pixels, target_color.expand_as(pixels))

    yellow_loss = masked_color_loss(yellow_mask, [0.95, 0.82, 0.10])
    red_loss = masked_color_loss(red_mask, [0.90, 0.10, 0.10])
    blue_loss = masked_color_loss(blue_mask, [0.10, 0.25, 0.90])
    background_loss = masked_color_loss(background_mask, [0.90, 0.90, 0.86])
    return 2.0 * yellow_loss + 1.5 * red_loss + blue_loss + 0.25 * background_loss


def create_ema_model(model: nn.Module):
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float):
    with torch.no_grad():
        for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
            ema_parameter.data.mul_(decay).add_(parameter.data, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.data.copy_(buffer.data)


def count_trainable_parameters(model: nn.Module):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def load_state_dict_from_checkpoint(path: Path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def save_training_state_checkpoint(
    path: Path,
    *,
    generator: nn.Module,
    discriminator: nn.Module,
    ema_generator: nn.Module | None,
    optimizer_g,
    optimizer_d,
    config,
    completed_steps: int,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "ema_generator": ema_generator.state_dict() if ema_generator is not None else None,
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "config": config,
            "completed_steps": completed_steps,
        },
        path,
    )


def load_training_state_checkpoint(path: Path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def set_optimizer_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def should_use_amp(config: TrainingConfig, device: torch.device) -> bool:
    return bool(config.amp and device.type == "cuda")


def scale_loss_for_accumulation(loss, grad_accum_steps: int):
    return loss / grad_accum_steps


def should_save_epoch_artifact(epoch: int, total_epochs: int, interval: int) -> bool:
    return epoch == 1 or epoch == total_epochs or epoch % interval == 0


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loss_axis_label(mode: str):
    if mode == "bce":
        return "BCEWithLogits loss"
    if mode == "lsgan":
        return "LSGAN MSE loss"
    raise ValueError(f"Unsupported adversarial loss mode: {mode}")


def save_loss_plot(history, output_path: Path, loss_mode: str = "bce"):
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["loss_d"], label="D loss")
    plt.plot(history["epoch"], history["loss_g"], label="G loss")
    plt.xlabel("Epoch")
    plt.ylabel(loss_axis_label(loss_mode))
    plt.title(f"GAN Training Loss ({loss_mode.upper()})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_history_csv(history, output_path: Path):
    fieldnames = [
        "epoch",
        "loss_d",
        "loss_d_total",
        "loss_d_real",
        "loss_d_fake",
        "loss_g",
        "d_real",
        "d_fake",
    ]
    fieldnames = [field for field in fieldnames if field in history]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, epoch in enumerate(history["epoch"]):
            writer.writerow({field: history[field][index] for field in fieldnames})


def _tail_average(values, tail: int):
    window = values[-min(tail, len(values)) :]
    return float(sum(window) / len(window))


def compute_diagnostics(history, tail=5):
    tail_loss_d = _tail_average(history["loss_d"], tail)
    tail_loss_g = _tail_average(history["loss_g"], tail)
    tail_d_real = _tail_average(history["d_real"], tail)
    tail_d_fake = _tail_average(history["d_fake"], tail)
    ratio = tail_loss_d / max(tail_loss_g, 1e-8)

    if tail_d_real >= 0.85 and tail_d_fake <= 0.08 and tail_loss_g > tail_loss_d:
        judgment = "discriminator_too_strong"
    elif tail_d_real <= 0.75 and tail_d_fake >= 0.45:
        judgment = "discriminator_too_weak"
    elif math.isfinite(ratio) and 0.4 <= ratio <= 2.5 and 0.15 <= tail_d_fake <= 0.45:
        judgment = "roughly_balanced_short_run"
    elif tail_loss_g > tail_loss_d * 3:
        judgment = "generator_struggling"
    else:
        judgment = "needs_visual_review"

    return {
        "tail_window": min(tail, len(history["loss_d"])),
        "tail_loss_d": tail_loss_d,
        "tail_loss_g": tail_loss_g,
        "tail_d_real": tail_d_real,
        "tail_d_fake": tail_d_fake,
        "tail_loss_d_over_loss_g": ratio,
        "stability_judgment": judgment,
    }


def train(config: TrainingConfig):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

    dataset_size = count_images(config.dataset_dir)
    if config.skip_prepare:
        prepared_count = len(prepared_image_paths(config.processed_dir))
        if prepared_count <= 0:
            raise ValueError(f"No prepared images found in {config.processed_dir}; cannot skip preprocessing")
    else:
        prepared_count = prepare_images(
            config.dataset_dir,
            config.processed_dir,
            config.image_size,
            min_target_anchor_fraction=config.min_target_anchor_fraction,
            target_crop_expansion=config.target_crop_expansion,
        )
    dataset = ArrowImageDataset(config.processed_dir, config.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        drop_last=False,
    )

    device = select_device()
    generator = Generator(
        image_size=config.image_size,
        latent_dim=config.latent_dim,
        channels=config.channels,
        features=config.generator_features,
        mode=config.generator_mode,
    ).to(device)
    discriminator = Discriminator(
        image_size=config.image_size,
        channels=config.channels,
        features=config.discriminator_features,
        norm=config.discriminator_norm,
    )
    if config.discriminator_mode == "patch":
        discriminator = PatchDiscriminator(
            image_size=config.image_size,
            channels=config.channels,
            features=config.discriminator_features,
            norm=config.discriminator_norm,
        )
    elif config.discriminator_mode != "global":
        raise ValueError(f"Unsupported discriminator mode: {config.discriminator_mode}")
    discriminator = discriminator.to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    ema_generator = create_ema_model(generator) if config.ema_decay > 0 else None
    if config.resume_generator is not None:
        generator.load_state_dict(load_state_dict_from_checkpoint(config.resume_generator, map_location=device))
    if config.resume_discriminator is not None:
        discriminator.load_state_dict(load_state_dict_from_checkpoint(config.resume_discriminator, map_location=device))
    if ema_generator is not None:
        if config.resume_ema_generator is not None:
            ema_generator.load_state_dict(load_state_dict_from_checkpoint(config.resume_ema_generator, map_location=device))
        else:
            ema_generator.load_state_dict(generator.state_dict())

    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=(config.beta1, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=config.g_lr, betas=(config.beta1, 0.999))
    if config.resume_training_state is not None:
        training_state = load_training_state_checkpoint(config.resume_training_state, map_location=device)
        generator.load_state_dict(training_state["generator"])
        discriminator.load_state_dict(training_state["discriminator"])
        if ema_generator is not None and training_state.get("ema_generator") is not None:
            ema_generator.load_state_dict(training_state["ema_generator"])
        optimizer_g.load_state_dict(training_state["optimizer_g"])
        optimizer_d.load_state_dict(training_state["optimizer_d"])
        set_optimizer_lr(optimizer_g, config.g_lr)
        set_optimizer_lr(optimizer_d, config.d_lr)
    fixed_noise = torch.randn(16, config.latent_dim, 1, 1, device=device)
    use_amp = should_use_amp(config, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_dir = config.output_dir / time.strftime("gan_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "epoch": [],
        "loss_d": [],
        "loss_d_total": [],
        "loss_d_real": [],
        "loss_d_fake": [],
        "loss_g": [],
        "d_real": [],
        "d_fake": [],
    }
    completed_steps = 0
    for epoch in range(1, config.epochs + 1):
        loss_d_values = []
        loss_d_total_values = []
        loss_d_real_values = []
        loss_d_fake_values = []
        loss_g_values = []
        d_real_values = []
        d_fake_values = []

        optimizer_d.zero_grad(set_to_none=True)
        optimizer_g.zero_grad(set_to_none=True)
        accumulated_batches = 0
        for batch_index, real_images in enumerate(dataloader, start=1):
            real_images = real_images.to(device)
            current_batch = real_images.size(0)

            with torch.cuda.amp.autocast(enabled=use_amp):
                real_for_discriminator = (
                    diff_augment(real_images, config.diffaugment_policy) if config.diffaugment else real_images
                )
                output_real = discriminator(real_for_discriminator)
                loss_real = adversarial_loss(output_real, config.real_label, config.adversarial_loss_mode)

                noise = torch.randn(current_batch, config.latent_dim, 1, 1, device=device)
                fake_images = generator(noise)
                fake_for_discriminator = (
                    diff_augment(fake_images.detach(), config.diffaugment_policy)
                    if config.diffaugment
                    else fake_images.detach()
                )
                output_fake = discriminator(fake_for_discriminator)
                loss_fake = adversarial_loss(output_fake, config.fake_label, config.adversarial_loss_mode)

                loss_d = loss_real + loss_fake
                scaled_loss_d = scale_loss_for_accumulation(loss_d, config.grad_accum_steps)
            scaler.scale(scaled_loss_d).backward()

            if config.freeze_discriminator_during_generator_step:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(False)
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake_for_generator = diff_augment(fake_images, config.diffaugment_policy) if config.diffaugment else fake_images
                output_for_generator = discriminator(fake_for_generator)
                loss_g = adversarial_loss(output_for_generator, config.real_label, config.adversarial_loss_mode)
                if config.target_prior_weight > 0:
                    loss_g = loss_g + config.target_prior_weight * target_ring_prior_loss(fake_images)
                scaled_loss_g = scale_loss_for_accumulation(loss_g, config.grad_accum_steps)
            scaler.scale(scaled_loss_g).backward()
            if config.freeze_discriminator_during_generator_step:
                for parameter in discriminator.parameters():
                    parameter.requires_grad_(True)

            loss_d_values.append(discriminator_log_loss(loss_real.detach(), loss_fake.detach()).item())
            loss_d_total_values.append(loss_d.item())
            loss_d_real_values.append(loss_real.item())
            loss_d_fake_values.append(loss_fake.item())
            loss_g_values.append(loss_g.item())
            d_real_values.append(discriminator_confidence(output_real, config.adversarial_loss_mode))
            d_fake_values.append(discriminator_confidence(output_fake, config.adversarial_loss_mode))
            completed_steps += 1
            accumulated_batches += 1
            reached_accumulation_boundary = accumulated_batches >= config.grad_accum_steps
            reached_epoch_end = batch_index == len(dataloader)
            reached_max_steps = config.max_steps is not None and completed_steps >= config.max_steps
            if reached_accumulation_boundary or reached_epoch_end or reached_max_steps:
                scaler.step(optimizer_d)
                scaler.step(optimizer_g)
                scaler.update()
                optimizer_d.zero_grad(set_to_none=True)
                optimizer_g.zero_grad(set_to_none=True)
                accumulated_batches = 0
                if ema_generator is not None:
                    update_ema_model(ema_generator, generator, config.ema_decay)
            if reached_max_steps:
                break

        history["epoch"].append(epoch)
        history["loss_d"].append(float(sum(loss_d_values) / len(loss_d_values)))
        history["loss_d_total"].append(float(sum(loss_d_total_values) / len(loss_d_total_values)))
        history["loss_d_real"].append(float(sum(loss_d_real_values) / len(loss_d_real_values)))
        history["loss_d_fake"].append(float(sum(loss_d_fake_values) / len(loss_d_fake_values)))
        history["loss_g"].append(float(sum(loss_g_values) / len(loss_g_values)))
        history["d_real"].append(float(sum(d_real_values) / len(d_real_values)))
        history["d_fake"].append(float(sum(d_fake_values) / len(d_fake_values)))

        if should_save_epoch_artifact(epoch, config.epochs, config.sample_interval):
            with torch.no_grad():
                sample_model = ema_generator if ema_generator is not None else generator
                sample_model.eval()
                sample_images = sample_model(fixed_noise).detach().cpu()
                vutils.save_image(sample_images, run_dir / f"samples_epoch_{epoch:03d}.png", normalize=True, nrow=4)
                generator.train()

        if should_save_epoch_artifact(epoch, config.epochs, config.checkpoint_interval):
            save_training_state_checkpoint(
                run_dir / "training_state_latest.pt",
                generator=generator,
                discriminator=discriminator,
                ema_generator=ema_generator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                config={key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                completed_steps=completed_steps,
            )

        print(
            f"epoch={epoch:03d} "
            f"loss_d={history['loss_d'][-1]:.4f} "
            f"loss_g={history['loss_g'][-1]:.4f} "
            f"d_real={history['d_real'][-1]:.4f} "
            f"d_fake={history['d_fake'][-1]:.4f}"
        )
        if config.max_steps is not None and completed_steps >= config.max_steps:
            break

    torch.save(generator.state_dict(), run_dir / "generator.pt")
    if ema_generator is not None:
        torch.save(ema_generator.state_dict(), run_dir / "generator_ema.pt")
    torch.save(discriminator.state_dict(), run_dir / "discriminator.pt")
    best_state_dict = ema_generator.state_dict() if ema_generator is not None else generator.state_dict()
    torch.save(
        {
            "state_dict": best_state_dict,
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
            "uses_ema": ema_generator is not None,
        },
        run_dir / "best_generator.pt",
    )
    save_training_state_checkpoint(
        run_dir / "training_state.pt",
        generator=generator,
        discriminator=discriminator,
        ema_generator=ema_generator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        config={key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        completed_steps=completed_steps,
    )
    save_loss_plot(history, run_dir / "loss_curve.png", config.adversarial_loss_mode)
    save_history_csv(history, run_dir / "history.csv")

    diagnostics = compute_diagnostics(history)
    final_loss_d = history["loss_d"][-1]
    final_loss_g = history["loss_g"][-1]
    balance_ratio = final_loss_d / max(final_loss_g, 1e-8)

    metrics = {
        "dataset_size": dataset_size,
        "prepared_count": prepared_count,
        "completed_steps": completed_steps,
        "device": str(device),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "parameter_counts": {
            "generator": count_trainable_parameters(generator),
            "discriminator": count_trainable_parameters(discriminator),
            "total": count_trainable_parameters(generator) + count_trainable_parameters(discriminator),
        },
        "final": {
            "loss_d": final_loss_d,
            "loss_d_total": history["loss_d_total"][-1],
            "loss_d_real": history["loss_d_real"][-1],
            "loss_d_fake": history["loss_d_fake"][-1],
            "loss_g": final_loss_g,
            "d_real": history["d_real"][-1],
            "d_fake": history["d_fake"][-1],
            "loss_d_over_loss_g": balance_ratio,
            "stability_judgment": diagnostics["stability_judgment"],
        },
        "diagnostics": diagnostics,
        "artifacts": {
            "run_dir": str(run_dir),
            "best_generator": str(run_dir / "best_generator.pt"),
            "generator_ema": str(run_dir / "generator_ema.pt") if ema_generator is not None else None,
            "training_state": str(run_dir / "training_state.pt"),
            "training_state_latest": str(run_dir / "training_state_latest.pt"),
            "loss_curve": str(run_dir / "loss_curve.png"),
            "history_csv": str(run_dir / "history.csv"),
            "last_samples": str(run_dir / f"samples_epoch_{history['epoch'][-1]:03d}.png"),
        },
    }
    with (run_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small-data DCGAN on arrow HEIC images.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset/origin_data"))
    parser.add_argument("--processed-dir", type=Path, default=Path("dataset/generate_data/processed_256"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--d-lr", type=float, default=None)
    parser.add_argument("--g-lr", type=float, default=None)
    parser.add_argument("--discriminator-norm", choices=["none", "instance", "batch"], default=None)
    parser.add_argument("--discriminator-mode", choices=["global", "patch"], default=None)
    parser.add_argument("--adversarial-loss-mode", choices=["bce", "lsgan"], default=None)
    parser.add_argument("--generator-features", type=int, default=None)
    parser.add_argument("--discriminator-features", type=int, default=None)
    parser.add_argument("--generator-mode", choices=["upsample", "transpose"], default=None)
    parser.add_argument("--diffaugment", action="store_true")
    parser.add_argument("--diffaugment-policy", default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--target-prior-weight", type=float, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--sample-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--no-freeze-discriminator-during-generator-step", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-generator", type=Path, default=None)
    parser.add_argument("--resume-discriminator", type=Path, default=None)
    parser.add_argument("--resume-ema-generator", type=Path, default=None)
    parser.add_argument("--resume-training-state", type=Path, default=None)
    parser.add_argument("--min-target-anchor-fraction", type=float, default=None)
    parser.add_argument("--target-crop-expansion", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_size = count_images(args.dataset_dir)
    overrides = {
        "dataset_dir": args.dataset_dir,
        "processed_dir": args.processed_dir,
        "output_dir": args.output_dir,
        "image_size": args.image_size,
        "seed": args.seed,
    }
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.d_lr is not None:
        overrides["d_lr"] = args.d_lr
    if args.g_lr is not None:
        overrides["g_lr"] = args.g_lr
    if args.discriminator_norm is not None:
        overrides["discriminator_norm"] = args.discriminator_norm
    if args.discriminator_mode is not None:
        overrides["discriminator_mode"] = args.discriminator_mode
    if args.adversarial_loss_mode is not None:
        overrides["adversarial_loss_mode"] = args.adversarial_loss_mode
    if args.generator_features is not None:
        overrides["generator_features"] = args.generator_features
    if args.discriminator_features is not None:
        overrides["discriminator_features"] = args.discriminator_features
    if args.generator_mode is not None:
        overrides["generator_mode"] = args.generator_mode
    if args.diffaugment:
        overrides["diffaugment"] = True
    if args.diffaugment_policy is not None:
        overrides["diffaugment_policy"] = args.diffaugment_policy
    if args.ema_decay is not None:
        overrides["ema_decay"] = args.ema_decay
    if args.target_prior_weight is not None:
        overrides["target_prior_weight"] = args.target_prior_weight
    if args.amp:
        overrides["amp"] = True
    if args.grad_accum_steps is not None:
        overrides["grad_accum_steps"] = args.grad_accum_steps
    if args.sample_interval is not None:
        overrides["sample_interval"] = args.sample_interval
    if args.checkpoint_interval is not None:
        overrides["checkpoint_interval"] = args.checkpoint_interval
    if args.no_freeze_discriminator_during_generator_step:
        overrides["freeze_discriminator_during_generator_step"] = False
    if args.skip_prepare:
        overrides["skip_prepare"] = True
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.resume_generator is not None:
        overrides["resume_generator"] = args.resume_generator
    if args.resume_discriminator is not None:
        overrides["resume_discriminator"] = args.resume_discriminator
    if args.resume_ema_generator is not None:
        overrides["resume_ema_generator"] = args.resume_ema_generator
    if args.resume_training_state is not None:
        overrides["resume_training_state"] = args.resume_training_state
    if args.min_target_anchor_fraction is not None:
        overrides["min_target_anchor_fraction"] = args.min_target_anchor_fraction
    if args.target_crop_expansion is not None:
        overrides["target_crop_expansion"] = args.target_crop_expansion
    config = TrainingConfig.from_dataset_size(dataset_size, **overrides)
    metrics = train(config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
