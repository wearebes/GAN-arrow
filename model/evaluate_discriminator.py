import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model.generate_samples import load_generator
from model.train_gan import Discriminator, PatchDiscriminator, prepared_image_paths, select_device


PREPARED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _quantile(sorted_values, fraction: float):
    if not sorted_values:
        raise ValueError("Cannot compute quantile for empty values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return float(sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight)


def summarize_probabilities(probabilities, fake_expected: bool):
    values = [float(value) for value in probabilities]
    if not values:
        raise ValueError("probabilities must not be empty")
    sorted_values = sorted(values)
    if fake_expected:
        correct = sum(value < 0.5 for value in values)
    else:
        correct = sum(value >= 0.5 for value in values)
    return {
        "count": len(values),
        "mean_p_real": float(sum(values) / len(values)),
        "median_p_real": _quantile(sorted_values, 0.5),
        "min_p_real": float(sorted_values[0]),
        "q25_p_real": _quantile(sorted_values, 0.25),
        "q75_p_real": _quantile(sorted_values, 0.75),
        "max_p_real": float(sorted_values[-1]),
        "correct_at_0_5": int(correct),
        "accuracy_at_0_5": float(correct / len(values)),
    }


class ImagePathDataset(Dataset):
    def __init__(self, paths, image_size: int):
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("No images found for discriminator evaluation")
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def image_paths(image_dir: Path):
    return sorted(
        path
        for path in Path(image_dir).iterdir()
        if path.is_file() and path.suffix.lower() in PREPARED_IMAGE_EXTENSIONS
    )


def _checkpoint_config(metrics_path: Path | None, args):
    if metrics_path is None:
        return {
            "image_size": args.image_size,
            "channels": args.channels,
            "discriminator_features": args.discriminator_features,
            "discriminator_norm": args.discriminator_norm,
            "discriminator_mode": args.discriminator_mode,
        }
    metrics = json.loads(metrics_path.read_text())
    config = metrics.get("config", {})
    return {
        "image_size": int(config.get("image_size", args.image_size)),
        "channels": int(config.get("channels", args.channels)),
        "discriminator_features": int(config.get("discriminator_features", args.discriminator_features)),
        "discriminator_norm": config.get("discriminator_norm", args.discriminator_norm),
        "discriminator_mode": config.get("discriminator_mode", args.discriminator_mode),
    }


def load_discriminator(checkpoint_path: Path, config: dict, device):
    if config["discriminator_mode"] == "patch":
        discriminator = PatchDiscriminator(
            image_size=config["image_size"],
            channels=config["channels"],
            features=config["discriminator_features"],
            norm=config["discriminator_norm"],
        )
    else:
        discriminator = Discriminator(
            image_size=config["image_size"],
            channels=config["channels"],
            features=config["discriminator_features"],
            norm=config["discriminator_norm"],
        )
    discriminator.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    discriminator.to(device)
    discriminator.eval()
    return discriminator


def _output_to_probabilities(output):
    probabilities = torch.sigmoid(output.detach())
    if probabilities.dim() > 1:
        probabilities = probabilities.mean(dim=1)
    return probabilities.cpu().tolist()


@torch.no_grad()
def evaluate_image_paths(discriminator, paths, image_size: int, device, batch_size=16):
    dataloader = DataLoader(ImagePathDataset(paths, image_size), batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities = []
    for images in dataloader:
        output = discriminator(images.to(device))
        probabilities.extend(_output_to_probabilities(output))
    return probabilities


@torch.no_grad()
def evaluate_fresh_generator(discriminator, generator_checkpoint: Path, device, num_samples=64, seed=42, batch_size=16):
    generator, config = load_generator(generator_checkpoint, device=device)
    torch.manual_seed(seed)
    probabilities = []
    generated_count = 0
    while generated_count < num_samples:
        current_batch = min(batch_size, num_samples - generated_count)
        noise = torch.randn(current_batch, config["latent_dim"], 1, 1, device=device)
        images = generator(noise)
        output = discriminator(images)
        probabilities.extend(_output_to_probabilities(output))
        generated_count += current_batch
    return probabilities


def evaluate_discriminator(
    discriminator_checkpoint: Path,
    metrics_path: Path | None,
    real_dir: Path | None,
    generated_dir: Path | None,
    generator_checkpoint: Path | None,
    output_path: Path,
    num_fresh=64,
    seed=42,
    batch_size=16,
    args=None,
):
    device = select_device()
    config = _checkpoint_config(metrics_path, args or argparse.Namespace())
    discriminator = load_discriminator(discriminator_checkpoint, config, device)
    result = {
        "checkpoint": str(discriminator_checkpoint),
        "device": str(device),
        "meaning": "probabilities are sigmoid(D output), i.e. P(real) according to discriminator",
        "config": config,
    }
    if real_dir is not None:
        real_paths = prepared_image_paths(real_dir)
        real_probabilities = evaluate_image_paths(discriminator, real_paths, config["image_size"], device, batch_size)
        result["real"] = summarize_probabilities(real_probabilities, fake_expected=False)
    if generated_dir is not None:
        generated_probabilities = evaluate_image_paths(
            discriminator, image_paths(generated_dir), config["image_size"], device, batch_size
        )
        result["generated_saved"] = summarize_probabilities(generated_probabilities, fake_expected=True)
    if generator_checkpoint is not None and num_fresh > 0:
        fresh_probabilities = evaluate_fresh_generator(
            discriminator,
            generator_checkpoint,
            device,
            num_samples=num_fresh,
            seed=seed,
            batch_size=batch_size,
        )
        result["generated_fresh"] = summarize_probabilities(fresh_probabilities, fake_expected=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate discriminator P(real) on real and generated images.")
    parser.add_argument("--discriminator-checkpoint", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--real-dir", type=Path, default=None)
    parser.add_argument("--generated-dir", type=Path, default=None)
    parser.add_argument("--generator-checkpoint", type=Path, default=None)
    parser.add_argument("--num-fresh", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--discriminator-features", type=int, default=32)
    parser.add_argument("--discriminator-norm", choices=["none", "instance", "batch"], default="none")
    parser.add_argument("--discriminator-mode", choices=["global", "patch"], default="global")
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate_discriminator(
        discriminator_checkpoint=args.discriminator_checkpoint,
        metrics_path=args.metrics,
        real_dir=args.real_dir,
        generated_dir=args.generated_dir,
        generator_checkpoint=args.generator_checkpoint,
        output_path=args.out,
        num_fresh=args.num_fresh,
        seed=args.seed,
        batch_size=args.batch_size,
        args=args,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
