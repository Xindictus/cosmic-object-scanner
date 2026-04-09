"""Unit tests for models.visualization module."""

from unittest.mock import patch

import pytest
import torch

from cosmic_object_scanner.models.visualization import visualize_prediction_v2


@pytest.mark.unit
class TestVisualizePredictionV2:
    """Tests for Faster R-CNN visualization."""

    @patch("cosmic_object_scanner.models.visualization.plt")
    def test_runs_without_error(self, mock_plt: object) -> None:
        import numpy as np

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        prediction = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        # Should not raise
        visualize_prediction_v2(image, prediction, threshold=0.5)

    @patch("cosmic_object_scanner.models.visualization.plt")
    def test_custom_category_names(self, mock_plt: object) -> None:
        import numpy as np

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        prediction = [
            {
                "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
            }
        ]
        custom_names = {0: "CustomClass"}
        visualize_prediction_v2(image, prediction, threshold=0.5, category_names=custom_names)
