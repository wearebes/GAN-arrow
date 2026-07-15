import unittest
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch.nn as nn
import torch
import yaml

from PIL import Image
import numpy as np

from model.train_gan import (
    ArrowImageDataset,
    Discriminator,
    Generator,
    PatchDiscriminator,
    TrainingConfig,
    adversarial_loss,
    compute_diagnostics,
    count_images,
    diff_augment,
    discriminator_log_loss,
    image_has_signal,
    load_state_dict_from_checkpoint,
    loss_axis_label,
    prepare_images,
    prepared_image_paths,
    scale_loss_for_accumulation,
    should_save_epoch_artifact,
    should_use_amp,
    target_ring_prior_loss,
    train,
    weights_init,
)


class GanTrainingConfigTests(unittest.TestCase):
    def test_count_images_includes_heic_files(self):
        dataset_dir = Path("dataset/origin_data")

        self.assertEqual(count_images(dataset_dir), 109)

    def test_small_dataset_defaults_avoid_discriminator_batchnorm(self):
        config = TrainingConfig.from_dataset_size(109)
        discriminator = Discriminator(image_size=config.image_size, norm=config.discriminator_norm)

        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.image_size, 256)
        self.assertEqual(config.processed_dir, Path("dataset/generate_data/processed_256"))
        self.assertEqual(config.discriminator_norm, "none")
        self.assertEqual(config.real_label, 0.9)
        self.assertEqual(config.d_lr, 0.0001)
        self.assertEqual(config.g_lr, 0.0002)
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in discriminator.modules()))

    def test_generator_and_discriminator_support_256_images(self):
        config = TrainingConfig.from_dataset_size(109)
        generator = Generator(image_size=config.image_size, latent_dim=config.latent_dim, channels=config.channels)
        discriminator = Discriminator(image_size=config.image_size, channels=config.channels, norm=config.discriminator_norm)

        self.assertEqual(generator.output_size, 256)
        self.assertEqual(discriminator.input_size, 256)

    def test_256_large_global_model_exceeds_ten_million_parameters(self):
        generator = Generator(image_size=256, latent_dim=100, channels=3, features=64, mode="upsample")
        discriminator = Discriminator(image_size=256, channels=3, features=64, norm="none")

        total_parameters = sum(parameter.numel() for parameter in generator.parameters())
        total_parameters += sum(parameter.numel() for parameter in discriminator.parameters())

        self.assertGreaterEqual(total_parameters, 10_000_000)

    def test_training_config_can_enable_diffaugment_and_ema(self):
        config = TrainingConfig.from_dataset_size(
            109,
            diffaugment=True,
            ema_decay=0.999,
            skip_prepare=True,
            max_steps=1,
        )

        self.assertTrue(config.diffaugment)
        self.assertEqual(config.diffaugment_policy, "color,translation,cutout")
        self.assertAlmostEqual(config.ema_decay, 0.999)
        self.assertTrue(config.skip_prepare)
        self.assertEqual(config.max_steps, 1)

    def test_512_training_config_can_enable_memory_controls(self):
        config = TrainingConfig.from_dataset_size(
            109,
            image_size=512,
            processed_dir=Path("dataset/generate_data/processed_512_front"),
            generator_features=96,
            discriminator_features=32,
            batch_size=4,
            amp=True,
            grad_accum_steps=4,
            sample_interval=25,
            checkpoint_interval=25,
            freeze_discriminator_during_generator_step=True,
        )

        self.assertEqual(config.image_size, 512)
        self.assertEqual(config.processed_dir, Path("dataset/generate_data/processed_512_front"))
        self.assertEqual(config.generator_features, 96)
        self.assertEqual(config.discriminator_features, 32)
        self.assertEqual(config.batch_size, 4)
        self.assertTrue(config.amp)
        self.assertEqual(config.grad_accum_steps, 4)
        self.assertEqual(config.sample_interval, 25)
        self.assertEqual(config.checkpoint_interval, 25)
        self.assertTrue(config.freeze_discriminator_during_generator_step)

    def test_amp_only_enables_on_cuda(self):
        config = TrainingConfig.from_dataset_size(109, amp=True)

        self.assertFalse(should_use_amp(config, torch.device("cpu")))
        self.assertTrue(should_use_amp(config, torch.device("cuda")))
        self.assertFalse(should_use_amp(TrainingConfig.from_dataset_size(109, amp=False), torch.device("cuda")))

    def test_gradient_accumulation_scales_loss(self):
        loss = torch.tensor(8.0)

        self.assertEqual(scale_loss_for_accumulation(loss, 4).item(), 2.0)

    def test_should_save_epoch_artifact_respects_interval(self):
        self.assertTrue(should_save_epoch_artifact(epoch=1, total_epochs=100, interval=25))
        self.assertFalse(should_save_epoch_artifact(epoch=24, total_epochs=100, interval=25))
        self.assertTrue(should_save_epoch_artifact(epoch=25, total_epochs=100, interval=25))
        self.assertTrue(should_save_epoch_artifact(epoch=100, total_epochs=100, interval=25))

    def test_diffaugment_preserves_image_shape_and_gradients(self):
        images = torch.randn(2, 3, 256, 256, requires_grad=True)

        augmented = diff_augment(images, policy="color,translation,cutout")
        augmented.mean().backward()

        self.assertEqual(tuple(augmented.shape), (2, 3, 256, 256))
        self.assertIsNotNone(images.grad)

    def test_target_ring_prior_prefers_centered_colored_rings(self):
        yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
        radius = torch.sqrt((xx - 128) ** 2 + (yy - 128) ** 2)
        target = torch.ones(1, 3, 256, 256)
        target[:, :, radius < 70] = torch.tensor([0.1, 0.25, 0.9]).view(1, 3, 1)
        target[:, :, radius < 46] = torch.tensor([0.9, 0.1, 0.1]).view(1, 3, 1)
        target[:, :, radius < 22] = torch.tensor([0.95, 0.82, 0.1]).view(1, 3, 1)
        target = target * 2 - 1
        noise = torch.zeros_like(target)

        self.assertLess(target_ring_prior_loss(target).item(), target_ring_prior_loss(noise).item())

    def test_generator_uses_named_standard_blocks(self):
        generator = Generator(image_size=256, latent_dim=100, channels=3, features=16, mode="upsample")

        self.assertIsInstance(generator.blocks, nn.ModuleList)
        self.assertTrue(hasattr(generator, "project"))
        self.assertTrue(hasattr(generator, "to_rgb"))
        self.assertGreater(len(generator.blocks), 0)

    def test_weights_init_handles_named_model_blocks(self):
        generator = Generator(image_size=256, latent_dim=100, channels=3, features=16, mode="upsample")

        generator.apply(weights_init)

        self.assertEqual(generator.output_size, 256)

    def test_discriminators_use_named_standard_blocks(self):
        global_discriminator = Discriminator(image_size=256, channels=3, features=16, norm="none")
        patch_discriminator = PatchDiscriminator(image_size=256, channels=3, features=16, norm="none")

        self.assertIsInstance(global_discriminator.blocks, nn.ModuleList)
        self.assertTrue(hasattr(global_discriminator, "from_rgb"))
        self.assertTrue(hasattr(global_discriminator, "classifier"))
        self.assertIsInstance(patch_discriminator.blocks, nn.ModuleList)
        self.assertTrue(hasattr(patch_discriminator, "patch_head"))

    def test_patch_discriminator_outputs_multiple_local_scores(self):
        discriminator = PatchDiscriminator(image_size=256, channels=3, features=16, norm="none")

        output = discriminator(torch.randn(2, 3, 256, 256))

        self.assertEqual(output.shape[0], 2)
        self.assertGreater(output.shape[1], 1)

    def test_lsgan_loss_targets_match_patch_output_shape(self):
        output = torch.zeros(2, 16)

        loss = adversarial_loss(output, 0.9, mode="lsgan")

        self.assertGreater(loss.item(), 0)

    def test_default_generator_uses_upsample_not_transposed_conv(self):
        config = TrainingConfig.from_dataset_size(109)
        generator = Generator(
            image_size=config.image_size,
            latent_dim=config.latent_dim,
            channels=config.channels,
            mode=config.generator_mode,
        )

        self.assertEqual(config.generator_mode, "upsample")
        self.assertFalse(any(isinstance(module, nn.ConvTranspose2d) for module in generator.modules()))

    def test_upsample_generator_accepts_training_noise_shape(self):
        config = TrainingConfig.from_dataset_size(109)
        generator = Generator(
            image_size=config.image_size,
            latent_dim=config.latent_dim,
            channels=config.channels,
            mode=config.generator_mode,
        )

        output = generator(torch.randn(2, config.latent_dim, 1, 1))

        self.assertEqual(tuple(output.shape), (2, 3, 256, 256))

    def test_transpose_generator_accepts_training_noise_shape(self):
        config = TrainingConfig.from_dataset_size(109)
        generator = Generator(
            image_size=128,
            latent_dim=config.latent_dim,
            channels=config.channels,
            mode="transpose",
        )

        output = generator(torch.randn(2, config.latent_dim, 1, 1))

        self.assertEqual(tuple(output.shape), (2, 3, 128, 128))

    def test_tail_diagnostics_detects_discriminator_too_strong(self):
        diagnostics = compute_diagnostics(
            {
                "loss_d": [1.2, 0.9, 0.4, 0.35, 0.33],
                "loss_g": [1.0, 2.0, 5.5, 7.0, 8.0],
                "d_real": [0.7, 0.8, 0.91, 0.92, 0.93],
                "d_fake": [0.4, 0.2, 0.02, 0.01, 0.01],
            },
            tail=3,
        )

        self.assertEqual(diagnostics["stability_judgment"], "discriminator_too_strong")
        self.assertLess(diagnostics["tail_d_fake"], 0.05)

    def test_discriminator_logged_loss_averages_real_and_fake_terms(self):
        loss_real = torch.tensor(0.6)
        loss_fake = torch.tensor(0.2)

        logged_loss = discriminator_log_loss(loss_real, loss_fake)

        self.assertAlmostEqual(logged_loss.item(), 0.4)

    def test_lsgan_loss_plot_label_does_not_claim_bce(self):
        self.assertEqual(loss_axis_label("lsgan"), "LSGAN MSE loss")
        self.assertEqual(loss_axis_label("bce"), "BCEWithLogits loss")

    def test_prepare_images_writes_non_black_lossless_256_cache(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "processed_256"
            prepare_images(Path("dataset/origin_data"), output_dir, 256, limit=1)
            cached = next(output_dir.glob("*.png"))

            self.assertTrue(image_has_signal(cached))
            with Image.open(cached) as image:
                self.assertGreaterEqual(max(image.size), 256)
            self.assertEqual(cached.suffix, ".png")

    def test_prepared_image_paths_prefer_lossless_png_over_stale_jpg(self):
        with TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source"
            source_dir.mkdir()
            Image.new("RGB", (256, 256), (255, 0, 0)).save(source_dir / "sample.jpg", format="JPEG")
            Image.new("RGB", (256, 256), (0, 255, 0)).save(source_dir / "sample.png", format="PNG")

            paths = prepared_image_paths(source_dir)
            dataset = ArrowImageDataset(source_dir, 256)

            self.assertEqual(paths, [source_dir / "sample.png"])
            self.assertEqual(dataset.paths, [source_dir / "sample.png"])

    def test_prepare_images_centers_colored_target_roi(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "processed_target_256"
            prepare_images(Path("dataset/origin_data"), output_dir, 256, limit=1)
            cached = next(output_dir.glob("*.png"))

            with Image.open(cached) as image:
                resized = image.convert("RGB").resize((256, 256))
            array = np.asarray(resized)
            center = array[76:180, 76:180]
            max_channel = center.max(axis=2)
            min_channel = center.min(axis=2)
            saturated_fraction = ((array.max(axis=2) - array.min(axis=2)) > 50).mean()

            self.assertGreater(saturated_fraction, 0.21)

    def test_prepare_images_skips_photos_without_target_anchor(self):
        for filename in ("IMG_4024.HEIC",):
            with self.subTest(filename=filename), TemporaryDirectory() as temp_dir:
                temp_origin = Path(temp_dir) / "origin"
                temp_origin.mkdir()
                shutil.copy(Path("dataset/origin_data") / filename, temp_origin / filename)
                output_dir = Path(temp_dir) / "processed_target_256"

                prepared_count = prepare_images(temp_origin, output_dir, 256)

                self.assertEqual(prepared_count, 0)
                self.assertEqual(list(output_dir.glob("*.png")), [])

    def test_prepare_images_removes_stale_jpg_when_source_is_rejected(self):
        with TemporaryDirectory() as temp_dir:
            temp_origin = Path(temp_dir) / "origin"
            temp_origin.mkdir()
            shutil.copy(Path("dataset/origin_data/IMG_4024.HEIC"), temp_origin / "IMG_4024.HEIC")
            output_dir = Path(temp_dir) / "processed_target_256"
            output_dir.mkdir()
            Image.new("RGB", (256, 256), (255, 0, 0)).save(output_dir / "IMG_4024.jpg", format="JPEG")

            prepared_count = prepare_images(temp_origin, output_dir, 256)

            self.assertEqual(prepared_count, 0)
            self.assertEqual(prepared_image_paths(output_dir), [])

    def test_prepare_images_skips_non_target_nondefault_heic_streams(self):
        with TemporaryDirectory() as temp_dir:
            temp_origin = Path(temp_dir) / "origin"
            temp_origin.mkdir()
            shutil.copy(Path("dataset/origin_data/IMG_4057.HEIC"), temp_origin / "IMG_4057.HEIC")
            output_dir = Path(temp_dir) / "processed_target_256"

            prepared_count = prepare_images(temp_origin, output_dir, 256)

            self.assertEqual(prepared_count, 0)
            self.assertEqual(list(output_dir.glob("*.png")), [])

    def test_prepare_images_skips_undecodable_heic_files(self):
        with TemporaryDirectory() as temp_dir:
            temp_origin = Path(temp_dir) / "origin"
            temp_origin.mkdir()
            (temp_origin / "broken.HEIC").write_bytes(b"not a real heic")
            output_dir = Path(temp_dir) / "processed_target_256"

            prepared_count = prepare_images(temp_origin, output_dir, 256)

            self.assertEqual(prepared_count, 0)
            self.assertEqual(prepared_image_paths(output_dir), [])

    def test_train_reports_empty_preprocess_cache_cause(self):
        with TemporaryDirectory() as temp_dir:
            temp_origin = Path(temp_dir) / "origin"
            temp_origin.mkdir()
            (temp_origin / "broken.HEIC").write_bytes(b"not a real heic")
            config = TrainingConfig(
                dataset_dir=temp_origin,
                processed_dir=Path(temp_dir) / "processed",
                output_dir=Path(temp_dir) / "outputs",
                batch_size=1,
                epochs=1,
            )

            with self.assertRaisesRegex(ValueError, "decode_failed=1"):
                train(config)

    def test_prepare_images_keeps_clear_target_subset(self):
        with TemporaryDirectory() as temp_dir:
            temp_origin = Path(temp_dir) / "origin"
            temp_origin.mkdir()
            for filename in ("IMG_4014.HEIC", "IMG_4015.HEIC"):
                shutil.copy(Path("dataset/origin_data") / filename, temp_origin / filename)
            output_dir = Path(temp_dir) / "processed_front_256"

            prepared_count = prepare_images(temp_origin, output_dir, 256, min_target_anchor_fraction=0.06)

            self.assertEqual(prepared_count, 2)
            self.assertEqual(
                sorted(path.name for path in output_dir.glob("*.png")),
                ["IMG_4014.png", "IMG_4015.png"],
            )

    def test_prepare_images_can_make_tighter_target_core_crop(self):
        with TemporaryDirectory() as temp_dir:
            temp_origin = Path(temp_dir) / "origin"
            temp_origin.mkdir()
            shutil.copy(Path("dataset/origin_data/IMG_4014.HEIC"), temp_origin / "IMG_4014.HEIC")
            default_dir = Path(temp_dir) / "default"
            core_dir = Path(temp_dir) / "core"

            prepare_images(temp_origin, default_dir, 256, target_crop_expansion=2.9)
            prepare_images(temp_origin, core_dir, 256, target_crop_expansion=1.1)

            def anchor_fraction(path):
                with Image.open(path) as image:
                    hsv = np.asarray(image.convert("RGB").resize((256, 256)).convert("HSV"))
                hue = hsv[:, :, 0].astype(np.int16)
                saturation = hsv[:, :, 1].astype(np.int16)
                value = hsv[:, :, 2].astype(np.int16)
                strong = (saturation > 85) & (value > 90)
                red = ((hue < 12) | (hue > 242)) & strong
                yellow = (hue >= 24) & (hue <= 48) & strong
                return float((red | yellow).mean())

            default_fraction = anchor_fraction(next(default_dir.glob("*.png")))
            core_fraction = anchor_fraction(next(core_dir.glob("*.png")))

            self.assertGreater(core_fraction, default_fraction * 1.2)

    def test_matrix_runner_loads_yaml_and_writes_summary(self):
        from model.run_matrix import run_matrix

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            matrix_path = root / "matrix.yaml"
            output_dir = root / "matrix_outputs"
            matrix_path.write_text(
                yaml.safe_dump(
                    {
                        "epochs": 18,
                        "defaults": {
                            "batch_size": 16,
                            "generator_features": 16,
                            "discriminator_features": 16,
                        },
                        "experiments": [
                            {
                                "name": "normNone_dlr1e-4_glr2e-4",
                                "discriminator_norm": "none",
                                "d_lr": 0.0001,
                                "g_lr": 0.0002,
                            }
                        ],
                    }
                )
            )

            def fake_train(config):
                self.assertEqual(config.batch_size, 16)
                self.assertEqual(config.generator_features, 16)
                self.assertEqual(config.discriminator_features, 16)
                return {
                    "final": {"loss_d": 1.0, "loss_g": 0.8},
                    "diagnostics": {"stability_judgment": "roughly_balanced_short_run"},
                    "artifacts": {"run_dir": str(config.output_dir / "gan_fake")},
                }

            with patch("model.run_matrix.train", side_effect=fake_train):
                summary_path = run_matrix(
                    matrix_path=matrix_path,
                    dataset_dir=Path("dataset/origin_data"),
                    processed_dir=Path("dataset/generate_data/processed_256"),
                    output_dir=output_dir,
                )

            summary = summary_path.read_text()
            self.assertIn("normNone_dlr1e-4_glr2e-4", summary)
            self.assertIn("roughly_balanced_short_run", summary)

    def test_augmented_generator_writes_samples_and_grid(self):
        from model.generate_augmented import generate_augmented

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            output_dir = root / "augmented"
            source_dir.mkdir()
            Image.new("RGB", (256, 256), (220, 220, 220)).save(source_dir / "sample.jpg")

            metrics = generate_augmented(source_dir, output_dir, count=3, image_size=256, seed=1)

            self.assertEqual(metrics["source_count"], 1)
            self.assertEqual(metrics["generated_count"], 3)
            self.assertTrue((output_dir / "augmented_grid.png").exists())
            self.assertEqual(len(list((output_dir / "samples").glob("*.jpg"))), 3)

    def test_procedural_target_generator_writes_clear_256_targets(self):
        from model.generate_targets import generate_targets

        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "targets"

            metrics = generate_targets(output_dir, count=2, image_size=256, seed=1)

            self.assertEqual(metrics["generated_count"], 2)
            self.assertTrue((output_dir / "target_grid.png").exists())
            sample = output_dir / "samples" / "target_001.jpg"
            with Image.open(sample) as image:
                self.assertEqual(image.size, (256, 256))
                hsv = np.asarray(image.convert("HSV"))
            hue = hsv[:, :, 0].astype(np.int16)
            saturation = hsv[:, :, 1].astype(np.int16)
            value = hsv[:, :, 2].astype(np.int16)
            center = np.indices((256, 256))
            radius = np.sqrt((center[1] - 128) ** 2 + (center[0] - 128) ** 2)
            strong = (saturation > 70) & (value > 80)
            yellow = (hue >= 24) & (hue <= 48) & strong & (radius < 42)
            red = ((hue < 12) | (hue > 242)) & strong & (radius < 70)
            blue = (hue >= 125) & (hue <= 170) & strong & (radius < 98)
            dark = (value < 80) & (radius < 120)

            self.assertGreater(yellow.mean(), 0.01)
            self.assertGreater(red.mean(), 0.02)
            self.assertGreater(blue.mean(), 0.02)
            self.assertGreater(dark.mean(), 0.02)

    def test_target_scorer_accepts_synthetic_target_and_rejects_gray_image(self):
        from model.score_target import score_image

        with TemporaryDirectory() as temp_dir:
            target_path = Path(temp_dir) / "target.png"
            gray_path = Path(temp_dir) / "gray.png"
            yy, xx = np.indices((256, 256))
            radius = np.sqrt((xx - 128) ** 2 + (yy - 128) ** 2)
            image = np.full((256, 256, 3), 245, dtype=np.uint8)
            image[radius < 66] = (30, 80, 190)
            image[radius < 46] = (230, 30, 30)
            image[radius < 22] = (245, 210, 25)
            Image.fromarray(image).save(target_path)
            Image.new("RGB", (256, 256), (128, 128, 128)).save(gray_path)

            self.assertTrue(score_image(target_path)["passed"])
            self.assertFalse(score_image(gray_path)["passed"])

    def test_generate_samples_loads_generator_checkpoint_and_writes_256_outputs(self):
        from model.generate_samples import generate_samples

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "checkpoint.pt"
            output_dir = root / "samples"
            generator = Generator(image_size=256, latent_dim=100, channels=3, features=4, mode="upsample")
            torch.save(
                {
                    "state_dict": generator.state_dict(),
                    "config": {
                        "image_size": 256,
                        "latent_dim": 100,
                        "channels": 3,
                        "generator_features": 4,
                        "generator_mode": "upsample",
                    },
                },
                checkpoint_path,
            )

            metrics = generate_samples(checkpoint_path, output_dir, num_samples=4, seed=7)

            self.assertEqual(metrics["generated_count"], 4)
            self.assertTrue((output_dir / "contact_sheet.png").exists())
            self.assertEqual(len(list((output_dir / "samples").glob("*.png"))), 4)
            with Image.open(output_dir / "samples" / "sample_001.png") as image:
                self.assertEqual(image.size, (256, 256))

    def test_checkpoint_state_loader_accepts_wrapped_and_plain_state_dicts(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            generator = Generator(image_size=256, latent_dim=100, channels=3, features=4, mode="upsample")
            plain_path = root / "plain.pt"
            wrapped_path = root / "wrapped.pt"
            torch.save(generator.state_dict(), plain_path)
            torch.save({"state_dict": generator.state_dict()}, wrapped_path)

            self.assertEqual(load_state_dict_from_checkpoint(plain_path).keys(), generator.state_dict().keys())
            self.assertEqual(load_state_dict_from_checkpoint(wrapped_path).keys(), generator.state_dict().keys())

    def test_training_state_checkpoint_roundtrip_contains_optimizers(self):
        from model.train_gan import load_training_state_checkpoint, save_training_state_checkpoint

        with TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "training_state.pt"
            generator = Generator(image_size=256, latent_dim=100, channels=3, features=4, mode="upsample")
            discriminator = Discriminator(image_size=256, channels=3, features=4, norm="none")
            ema_generator = Generator(image_size=256, latent_dim=100, channels=3, features=4, mode="upsample")
            optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
            optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

            save_training_state_checkpoint(
                checkpoint_path,
                generator=generator,
                discriminator=discriminator,
                ema_generator=ema_generator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                config={"image_size": 256},
                completed_steps=12,
            )
            state = load_training_state_checkpoint(checkpoint_path)

            self.assertIn("generator", state)
            self.assertIn("discriminator", state)
            self.assertIn("ema_generator", state)
            self.assertIn("optimizer_g", state)
            self.assertIn("optimizer_d", state)
            self.assertEqual(state["completed_steps"], 12)

    def test_optimizer_learning_rate_can_be_overridden_after_resume(self):
        from model.train_gan import set_optimizer_lr

        generator = Generator(image_size=256, latent_dim=100, channels=3, features=4, mode="upsample")
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

        set_optimizer_lr(optimizer, 0.00005)

        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.00005])

    def test_discriminator_probability_summary_handles_real_and_generated_inputs(self):
        from model.evaluate_discriminator import summarize_probabilities

        real_summary = summarize_probabilities([0.8, 0.9, 0.95], fake_expected=False)
        generated_summary = summarize_probabilities([0.1, 0.2, 0.4], fake_expected=True)

        self.assertAlmostEqual(real_summary["mean_p_real"], 0.8833333333333333)
        self.assertAlmostEqual(generated_summary["mean_p_real"], 0.23333333333333334)
        self.assertEqual(real_summary["accuracy_at_0_5"], 1.0)
        self.assertEqual(generated_summary["accuracy_at_0_5"], 1.0)
        self.assertEqual(real_summary["count"], 3)
        self.assertEqual(generated_summary["count"], 3)


if __name__ == "__main__":
    unittest.main()
