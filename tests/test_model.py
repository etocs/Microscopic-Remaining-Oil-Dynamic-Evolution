import unittest

import numpy as np
import torch

from microscopic_remaining_oil_dynamic_evolution.model import (
    FusionModel,
    PhysicalConstraintLoss,
    predict_with_model,
)
from microscopic_remaining_oil_dynamic_evolution.preprocessing import (
    dynamic_time_series_preprocess,
    static_image_preprocess,
)


class FusionModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_forward_shapes(self) -> None:
        model = FusionModel(static_channels=1, dynamic_features=5, physics_weight=0.1)
        static_volume = torch.randn(2, 1, 16, 32, 32)
        dynamic_sequence = torch.randn(2, 30, 5)
        permeability = torch.full((2, 1), 0.2)
        viscosity = torch.full((2, 1), 1.5)
        pressure_gradient = torch.full((2, 1), 0.8)

        outputs = model(
            static_volume=static_volume,
            dynamic_sequence=dynamic_sequence,
            permeability=permeability,
            viscosity=viscosity,
            pressure_gradient=pressure_gradient,
        )
        self.assertEqual(outputs["saturation"].shape, (2, 1))
        self.assertEqual(outputs["velocity"].shape, (2, 1))
        self.assertEqual(outputs["transition_prob"].shape, (2, 1))
        self.assertEqual(outputs["darcy_velocity"].shape, (2, 1))
        self.assertTrue(torch.all((outputs["saturation"] >= 0.0) & (outputs["saturation"] <= 1.0)))
        self.assertTrue(
            torch.all(
                (outputs["transition_prob"] >= 0.0)
                & (outputs["transition_prob"] <= 1.0)
            )
        )
        self.assertTrue(torch.all(outputs["velocity"] >= 0))

    def test_loss_combines_prediction_and_darcy(self) -> None:
        model = FusionModel(static_channels=1, dynamic_features=5, physics_weight=0.1)
        criterion = PhysicalConstraintLoss(lambda_darcy=0.1)
        static_volume = torch.randn(2, 1, 16, 32, 32)
        dynamic_sequence = torch.randn(2, 30, 5)
        permeability = torch.full((2, 1), 0.2)
        viscosity = torch.full((2, 1), 1.5)
        pressure_gradient = torch.full((2, 1), 0.8)

        targets = {
            "saturation": torch.zeros(2, 1),
            "velocity": torch.ones(2, 1) * 0.5,
            "transition_prob": torch.zeros(2, 1),
        }
        preds = model(
            static_volume=static_volume,
            dynamic_sequence=dynamic_sequence,
            permeability=permeability,
            viscosity=viscosity,
            pressure_gradient=pressure_gradient,
        )
        loss = criterion(
            preds, targets, permeability, viscosity, pressure_gradient
        )
        self.assertGreater(loss.item(), 0.0)

    def test_predict_with_model_adds_loss_when_targets_supplied(self) -> None:
        model = FusionModel(static_channels=1, dynamic_features=5, physics_weight=0.05)
        static_volume = torch.randn(1, 1, 8, 16, 16)
        dynamic_sequence = torch.randn(1, 30, 5)
        permeability = torch.ones(1, 1) * 0.1
        viscosity = torch.ones(1, 1)
        pressure_gradient = torch.ones(1, 1) * 0.5
        targets = {
            "saturation": torch.ones(1, 1) * 0.2,
            "velocity": torch.ones(1, 1) * 0.3,
            "transition_prob": torch.ones(1, 1) * 0.1,
        }
        outputs = predict_with_model(
            model,
            static_volume,
            dynamic_sequence,
            permeability,
            viscosity,
            pressure_gradient,
            targets=targets,
        )
        self.assertIn("loss", outputs)
        self.assertGreaterEqual(outputs["loss"].item(), 0.0)


class PreprocessTest(unittest.TestCase):
    def test_static_preprocess_returns_expected_shape(self) -> None:
        volume = np.random.rand(32, 32, 8).astype(np.float32)
        tensor = static_image_preprocess(volume, target_shape=(64, 64, 16))
        self.assertEqual(tensor.shape, (1, 16, 64, 64))

    def test_dynamic_preprocess_aligns_steps_and_scales(self) -> None:
        seq = np.stack([np.linspace(0, 1, 10), np.linspace(1, 0, 10)], axis=1)
        processed = dynamic_time_series_preprocess(seq, target_steps=30)
        self.assertEqual(processed.shape, (30, 2))
        self.assertTrue(torch.all(processed >= 0))
        self.assertTrue(torch.all(processed <= 1))


if __name__ == "__main__":
    unittest.main()
