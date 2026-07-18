import unittest
import csv
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch.nn as nn
import torch
import torchvision.transforms as transforms

from PIL import Image
import numpy as np

from model.train_gan import (
    ArrowImageDataset,
    Discriminator,
    Generator,
    TrainingConfig,
    compute_diagnostics,
    completed_epoch_from_training_state,
    count_images,
    diff_augment,
    discriminator_log_loss,
    image_has_signal,
    image_meets_minimum_resolution,
    load_history_csv,
    load_training_config,
    load_state_dict_from_checkpoint,
    prepare_images,
    prepared_image_paths,
    scale_loss_for_accumulation,
    should_save_epoch_artifact,
    should_stop_early,
    should_use_amp,
    target_structure_error,
    train,
    trim_training_log,
    update_early_stopping,
    weights_init,
)
from model.ada_augment import AdaBcgAugment, AdaController


class GanTrainingConfigTests(unittest.TestCase):
    def test_prepare_v1_dataset_keeps_capture_groups_and_writes_one_png_per_source(self):
        from model.prepare_v1_dataset import prepare_v1_dataset

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "origin"
            staging_dir = root / "staging"
            output_dir = root / "v1_1024"
            source_dir.mkdir()
            staging_dir.mkdir()
            yy, xx = np.indices((64, 64))
            radius = np.sqrt((xx - 32) ** 2 + (yy - 32) ** 2)
            for index in range(15):
                name = f"IMG_{4000 + index}"
                image = np.full((64, 64, 3), 245, dtype=np.uint8)
                image[radius < 25] = (30, 80, 190)
                image[radius < 17] = (230, 30, 30)
                image[radius < 8] = (245, 210, 25)
                Image.fromarray(image).save(source_dir / f"{name}.png")
                Image.fromarray(image).save(staging_dir / f"{name}.png")

            report = prepare_v1_dataset(
                source_dir,
                output_dir,
                image_size=64,
                test_count=5,
                val_count=5,
                capture_group_size=5,
                prepared_staging=staging_dir,
            )

            self.assertEqual(report["split_counts"], {"train": 5, "val": 5, "test": 5})
            self.assertTrue(report["validation"]["passed"])
            self.assertEqual(sum(1 for split in ("train", "val", "test") for _ in (output_dir / split).glob("*.png")), 15)
            with (output_dir / "manifest.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            group_splits = {}
            for row in rows:
                group_splits.setdefault(row["capture_group"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in group_splits.values()))

    def test_offline_augmentation_is_separate_and_traceable(self):
        from model.augment_dataset import augment_dataset

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "original"
            output_dir = root / "augmented"
            input_dir.mkdir()
            source_path = input_dir / "source.png"
            Image.new("RGB", (64, 64), (120, 80, 40)).save(source_path)
            source_before = source_path.read_bytes()

            summary = augment_dataset(input_dir, output_dir, seed=42)

            self.assertEqual(summary["source_count"], 1)
            self.assertEqual(summary["output_count"], 5)
            self.assertEqual(len(list(output_dir.glob("*.png"))), 5)
            self.assertTrue((output_dir / "_review" / "preview.png").exists())
            self.assertEqual(source_path.read_bytes(), source_before)
            self.assertTrue((output_dir / "augmentation_manifest.csv").exists())
            self.assertTrue((output_dir / "preprocessing_report.json").exists())

    def test_json_config_keeps_experiment_output_together(self):
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "output_dir": "experiments/EXP-GAN-1024-001-base",
                        "image_size": 1024,
                        "generator_features": 48,
                        "discriminator_features": 16,
                        "seed": 42,
                    }
                )
            )

            config = load_training_config(config_path)

            self.assertEqual(config.output_dir, Path("experiments/EXP-GAN-1024-001-base"))
            self.assertEqual(config.seed, 42)

    def test_count_images_includes_heic_files(self):
        dataset_dir = Path("dataset/origin_data")

        self.assertEqual(count_images(dataset_dir), 109)

    def test_small_dataset_defaults_avoid_discriminator_batchnorm(self):
        config = TrainingConfig.from_dataset_size(109)
        discriminator = Discriminator(image_size=config.image_size)

        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.image_size, 256)
        self.assertEqual(config.processed_dir, Path("dataset/generate_data/processed_256"))
        self.assertEqual(config.real_label, 0.9)
        self.assertEqual(config.d_lr, 0.0001)
        self.assertEqual(config.g_lr, 0.0002)
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in discriminator.modules()))

    def test_generator_and_discriminator_support_256_images(self):
        config = TrainingConfig.from_dataset_size(109)
        generator = Generator(image_size=config.image_size, latent_dim=config.latent_dim, channels=config.channels)
        discriminator = Discriminator(image_size=config.image_size, channels=config.channels)

        self.assertEqual(generator.output_size, 256)
        self.assertEqual(discriminator.input_size, 256)

    def test_256_large_global_model_exceeds_ten_million_parameters(self):
        generator = Generator(image_size=256, latent_dim=100, channels=3, features=64)
        discriminator = Discriminator(image_size=256, channels=3, features=64)

        total_parameters = sum(parameter.numel() for parameter in generator.parameters())
        total_parameters += sum(parameter.numel() for parameter in discriminator.parameters())

        self.assertGreaterEqual(total_parameters, 10_000_000)

    def test_1024_compact_model_stays_within_parameter_budget(self):
        generator = Generator(image_size=1024, latent_dim=100, channels=3, features=48)
        discriminator = Discriminator(image_size=1024, channels=3, features=16)

        generator_parameters = sum(parameter.numel() for parameter in generator.parameters())
        total_parameters = generator_parameters + sum(parameter.numel() for parameter in discriminator.parameters())

        self.assertLess(generator_parameters, 2_000_000)
        self.assertLess(total_parameters, 3_000_000)

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

    def test_training_config_can_select_paper_default_ada(self):
        config = TrainingConfig.from_dataset_size(
            109,
            augmentation_mode="ada",
            ada_augpipe="bgc",
            ada_target=0.6,
        )

        self.assertEqual(config.augmentation_mode, "ada")
        self.assertEqual(config.ada_augpipe, "bgc")
        self.assertEqual(config.ada_interval, 4)
        self.assertEqual(config.ada_kimg, 500.0)

    def test_dataset_has_no_real_only_random_flip(self):
        with TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir)
            Image.new("RGB", (64, 64), "white").save(image_dir / "sample.png")
            dataset = ArrowImageDataset(image_dir, 64)

            self.assertFalse(any(isinstance(item, transforms.RandomHorizontalFlip) for item in dataset.transform.transforms))

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

    def test_early_stopping_requires_minimum_epochs_and_patience(self):
        best, stale, improved = update_early_stopping(0.9, 1.0, 3, min_delta=0.01)
        self.assertTrue(improved)
        self.assertEqual(best, 0.9)
        self.assertEqual(stale, 0)

        best, stale, improved = update_early_stopping(0.895, best, stale, min_delta=0.01)
        self.assertFalse(improved)
        self.assertEqual(stale, 1)
        self.assertFalse(should_stop_early(799, 8, patience_evaluations=8, min_epochs=800))
        self.assertTrue(should_stop_early(800, 8, patience_evaluations=8, min_epochs=800))

    def test_diffaugment_preserves_image_shape_and_gradients(self):
        images = torch.randn(2, 3, 256, 256, requires_grad=True)

        augmented = diff_augment(images, policy="color,translation,cutout")
        augmented.mean().backward()

        self.assertEqual(tuple(augmented.shape), (2, 3, 256, 256))
        self.assertIsNotNone(images.grad)

    def test_ada_bgc_is_identity_at_zero_and_differentiable_when_enabled(self):
        pipe = AdaBcgAugment()
        images = torch.randn(2, 3, 64, 64, requires_grad=True)

        unchanged = pipe(images, 0.0)
        augmented = pipe(images, 1.0)
        augmented.mean().backward()

        self.assertTrue(torch.equal(unchanged, images))
        self.assertEqual(tuple(augmented.shape), (2, 3, 64, 64))
        self.assertIsNotNone(images.grad)

    def test_ada_controller_uses_real_logit_sign_target(self):
        controller = AdaController(target=0.6, interval=2, speed_kimg=1.0)

        self.assertIsNone(controller.observe(torch.ones(2)))
        update = controller.observe(torch.ones(2))

        self.assertAlmostEqual(update["rt"], 1.0)
        self.assertAlmostEqual(update["p"], 0.004)
        restored = AdaController.from_state_dict(controller.state_dict())
        self.assertAlmostEqual(restored.probability, controller.probability)

    def test_target_ring_prior_prefers_centered_colored_rings(self):
        yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing="ij")
        radius = torch.sqrt((xx - 128) ** 2 + (yy - 128) ** 2)
        target = torch.ones(1, 3, 256, 256)
        target[:, :, radius < 70] = torch.tensor([0.1, 0.25, 0.9]).view(1, 3, 1)
        target[:, :, radius < 46] = torch.tensor([0.9, 0.1, 0.1]).view(1, 3, 1)
        target[:, :, radius < 22] = torch.tensor([0.95, 0.82, 0.1]).view(1, 3, 1)
        target = target * 2 - 1
        noise = torch.zeros_like(target)

        self.assertLess(target_structure_error(target).item(), target_structure_error(noise).item())

    def test_generator_uses_named_standard_blocks(self):
        generator = Generator(image_size=256, latent_dim=100, channels=3, features=16)

        self.assertIsInstance(generator.blocks, nn.ModuleList)
        self.assertTrue(hasattr(generator, "project"))
        self.assertTrue(hasattr(generator, "to_rgb"))
        self.assertGreater(len(generator.blocks), 0)

    def test_weights_init_handles_named_model_blocks(self):
        generator = Generator(image_size=256, latent_dim=100, channels=3, features=16)

        generator.apply(weights_init)

        self.assertEqual(generator.output_size, 256)

    def test_discriminators_use_named_standard_blocks(self):
        discriminator = Discriminator(image_size=256, channels=3, features=16)

        self.assertIsInstance(discriminator.blocks, nn.ModuleList)
        self.assertTrue(hasattr(discriminator, "from_rgb"))
        self.assertTrue(hasattr(discriminator, "classifier"))

    def test_default_generator_uses_upsample_not_transposed_conv(self):
        config = TrainingConfig.from_dataset_size(109)
        generator = Generator(
            image_size=config.image_size,
            latent_dim=config.latent_dim,
            channels=config.channels,
        )

        self.assertFalse(any(isinstance(module, nn.ConvTranspose2d) for module in generator.modules()))

    def test_upsample_generator_accepts_training_noise_shape(self):
        config = TrainingConfig.from_dataset_size(109)
        generator = Generator(
            image_size=config.image_size,
            latent_dim=config.latent_dim,
            channels=config.channels,
        )

        output = generator(torch.randn(2, config.latent_dim, 1, 1))

        self.assertEqual(tuple(output.shape), (2, 3, 256, 256))

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

    def test_prepare_images_writes_non_black_lossless_256_cache(self):
        with TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "processed_256"
            prepare_images(Path("dataset/origin_data"), output_dir, 256, limit=1)
            cached = next(output_dir.glob("*.png"))

            self.assertTrue(image_has_signal(cached))
            with Image.open(cached) as image:
                self.assertEqual(image.size, (256, 256))
            self.assertEqual(cached.suffix, ".png")
            report = json.loads((output_dir / "preprocessing_report.json").read_text())
            self.assertEqual(report["training_resolution"], 256)
            self.assertEqual(report["accepted_count"], 1)

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

    def test_prepare_images_rejects_colored_pool_scene_without_target(self):
        with TemporaryDirectory() as temp_dir:
            temp_origin = Path(temp_dir) / "origin"
            temp_origin.mkdir()
            shutil.copy(Path("dataset/origin_data/IMG_4090.HEIC"), temp_origin / "IMG_4090.HEIC")
            output_dir = Path(temp_dir) / "processed_target_1024"

            prepared_count = prepare_images(temp_origin, output_dir, 1024)

            self.assertEqual(prepared_count, 0)
            report = json.loads((output_dir / "preprocessing_report.json").read_text())
            self.assertEqual(report["rejected_counts"]["insufficient_target_anchor"], 1)

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

    def test_resolution_gate_rejects_upsampling_low_resolution_inputs(self):
        with TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "small.png"
            Image.new("RGB", (640, 896), "white").save(image_path)

            self.assertFalse(image_meets_minimum_resolution(image_path, 1024))
            self.assertTrue(image_meets_minimum_resolution(image_path, 512))

    def test_one_step_training_keeps_all_core_artifacts_in_exact_output_dir(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            processed_dir = root / "processed"
            output_dir = root / "EXP-SMOKE"
            source_dir.mkdir()
            processed_dir.mkdir()
            Image.new("RGB", (64, 64), "white").save(source_dir / "source.png")
            Image.new("RGB", (64, 64), (128, 64, 32)).save(processed_dir / "prepared.png")
            config = TrainingConfig(
                dataset_dir=source_dir,
                processed_dir=processed_dir,
                output_dir=output_dir,
                image_size=64,
                latent_dim=8,
                batch_size=1,
                epochs=1,
                generator_features=4,
                discriminator_features=4,
                augmentation_mode="ada",
                ada_interval=1,
                ada_kimg=1.0,
                ada_p_initial=0.2,
                ema_decay=0.9,
                sample_interval=1,
                checkpoint_interval=1,
                skip_prepare=True,
                max_steps=1,
                seed=42,
            )

            metrics = train(config)

            self.assertEqual(metrics["artifacts"]["run_dir"], str(output_dir))
            for name in [
                "training_config.json",
                "environment.json",
                "dataset_manifest.csv",
                "training.log",
                "history.csv",
                "ada_history.csv",
                "metrics.json",
                "best_generator.pt",
                "final_generator.pt",
                "best_discriminator.pt",
                "training_state_epoch_001.pt",
                "samples_epoch_001.png",
            ]:
                self.assertTrue((output_dir / name).exists(), name)
            self.assertEqual(metrics["augmentation"]["mode"], "ada")
            self.assertEqual(metrics["augmentation"]["ada_updates"], 1)

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
            generator = Generator(image_size=256, latent_dim=100, channels=3, features=4)
            torch.save(
                {
                    "state_dict": generator.state_dict(),
                    "config": {
                        "image_size": 256,
                        "latent_dim": 100,
                        "channels": 3,
                        "generator_features": 4,
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
            generator = Generator(image_size=256, latent_dim=100, channels=3, features=4)
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
            generator = Generator(image_size=256, latent_dim=100, channels=3, features=4)
            discriminator = Discriminator(image_size=256, channels=3, features=4)
            ema_generator = Generator(image_size=256, latent_dim=100, channels=3, features=4)
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
                completed_epoch=3,
            )
            state = load_training_state_checkpoint(checkpoint_path)

            self.assertIn("generator", state)
            self.assertIn("discriminator", state)
            self.assertIn("ema_generator", state)
            self.assertIn("optimizer_g", state)
            self.assertIn("optimizer_d", state)
            self.assertEqual(state["completed_steps"], 12)
            self.assertEqual(state["completed_epoch"], 3)

    def test_old_training_state_infers_epoch_from_checkpoint_name(self):
        self.assertEqual(
            completed_epoch_from_training_state({}, Path("training_state_epoch_020.pt")),
            20,
        )

    def test_resume_history_and_log_are_trimmed_to_checkpoint_epoch(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            history_path = root / "history.csv"
            history_path.write_text(
                "epoch,loss_d,loss_d_total,loss_d_real,loss_d_fake,loss_g,d_real,d_fake\n"
                "79,0.3,0.3,0.1,0.2,2.0,0.8,0.1\n"
                "80,0.4,0.4,0.2,0.2,1.8,0.8,0.2\n"
                "81,0.5,0.5,0.2,0.3,1.6,0.7,0.3\n"
            )
            log_path = root / "training.log"
            log_path.write_text("epoch=079 loss_d=0.3\nepoch=080 loss_d=0.4\nepoch=081 loss_d=0.5\n")

            history = load_history_csv(history_path, max_epoch=80)
            trim_training_log(log_path, max_epoch=80)

            self.assertEqual(history["epoch"], [79, 80])
            self.assertEqual(history["loss_g"], [2.0, 1.8])
            self.assertEqual(log_path.read_text(), "epoch=079 loss_d=0.3\nepoch=080 loss_d=0.4\n")

    def test_optimizer_learning_rate_can_be_overridden_after_resume(self):
        from model.train_gan import set_optimizer_lr

        generator = Generator(image_size=256, latent_dim=100, channels=3, features=4)
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

        set_optimizer_lr(optimizer, 0.00005)

        self.assertEqual([group["lr"] for group in optimizer.param_groups], [0.00005])

    def test_discriminator_probability_summary_handles_real_and_generated_inputs(self):
        from model.evaluate_discriminator import classify_predictions, summarize_probabilities

        real_summary = summarize_probabilities([0.8, 0.9, 0.95], fake_expected=False)
        generated_summary = summarize_probabilities([0.1, 0.2, 0.4], fake_expected=True)

        self.assertAlmostEqual(real_summary["mean_p_real"], 0.8833333333333333)
        self.assertAlmostEqual(generated_summary["mean_p_real"], 0.23333333333333334)
        self.assertEqual(real_summary["accuracy_at_0_5"], 1.0)
        self.assertEqual(generated_summary["accuracy_at_0_5"], 1.0)
        self.assertEqual(real_summary["count"], 3)
        self.assertEqual(generated_summary["count"], 3)
        records = classify_predictions(
            [Path("real_correct.png"), Path("real_incorrect.png")],
            [0.8, 0.2],
            expected_label="real",
        )
        self.assertTrue(records[0]["correct"])
        self.assertFalse(records[1]["correct"])
        self.assertEqual(records[1]["predicted_label"], "fake")

    def test_discriminator_evaluation_reports_balanced_accuracy(self):
        from model.evaluate_discriminator import summarize_classification

        summary = summarize_classification(
            {"accuracy_at_0_5": 1.0},
            {"accuracy_at_0_5": 0.0},
        )

        self.assertEqual(summary["real_accuracy"], 1.0)
        self.assertEqual(summary["fake_accuracy"], 0.0)
        self.assertEqual(summary["balanced_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
