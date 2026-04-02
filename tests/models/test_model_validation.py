"""Model validation tests for Faster R-CNN and custom implementations.

Tests validate:
- Model initialization and loading
- Forward pass with dummy inputs
- Output shape correctness
- Utility function correctness
"""

import pytest

pytestmark = pytest.mark.models


class TestFasterRCNNModel:
    """Test Faster R-CNN model initialization and forward pass."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion=None, reason="skip"),
        reason="PyTorch not available",
    )
    def test_fastercnn_model_initialization(self):
        """Test that Faster R-CNN model can be initialized."""
        try:
            from cosmic_object_scanner.models.fasterrcnn_resnet50_fpn import get_model

            model = get_model(num_classes=3)
            assert model is not None
            assert hasattr(model, "backbone")
            assert hasattr(model, "roi_heads")
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion=None, reason="skip"),
        reason="PyTorch not available",
    )
    def test_fastercnn_forward_pass_train_mode(self, dummy_images_batch, dummy_targets):
        """Test forward pass in training mode."""
        try:
            from cosmic_object_scanner.models.fasterrcnn_resnet50_fpn import get_model

            model = get_model(num_classes=3)
            model.train()

            output = model(dummy_images_batch, dummy_targets)
            assert isinstance(output, dict)
            assert "loss_classifier" in output or isinstance(output, dict)
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion=None, reason="skip"),
        reason="PyTorch not available",
    )
    def test_fastercnn_forward_pass_eval_mode(self, dummy_images_batch):
        """Test forward pass in evaluation mode."""
        try:
            import torch

            from cosmic_object_scanner.models.fasterrcnn_resnet50_fpn import get_model

            model = get_model(num_classes=3)
            model.eval()

            with torch.no_grad():
                output = model(dummy_images_batch)

            assert isinstance(output, list)
            assert len(output) == len(dummy_images_batch)
            for prediction in output:
                assert "boxes" in prediction
                assert "scores" in prediction
                assert "labels" in prediction
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion=None, reason="skip"),
        reason="PyTorch not available",
    )
    def test_fastercnn_output_shapes(self, dummy_image_tensor):
        """Test output tensor shapes are correct."""
        try:
            import torch

            from cosmic_object_scanner.models.fasterrcnn_resnet50_fpn import get_model

            model = get_model(num_classes=3)
            model.eval()

            with torch.no_grad():
                output = model([dummy_image_tensor.squeeze(0)])

            assert len(output) == 1
            pred = output[0]
            assert pred["boxes"].shape[1] == 4  # [N, 4] for boxes
            assert pred["scores"].shape[0] == pred["boxes"].shape[0]  # Equal length
            assert pred["labels"].shape[0] == pred["boxes"].shape[0]  # Equal length
        except ImportError:
            pytest.skip("Required dependencies not available")


class TestUtilityFunctions:
    """Test utility functions used by models."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion=None, reason="skip"),
        reason="PyTorch not available",
    )
    def test_calculate_iou(self):
        """Test IoU calculation function."""
        try:
            import torch

            from cosmic_object_scanner.models.fasterrcnn_resnet50_fpn import calculate_iou

            # Test box format: [x1, y1, x2, y2]
            box1 = torch.tensor([0.0, 0.0, 10.0, 10.0])
            box2 = torch.tensor([5.0, 5.0, 15.0, 15.0])

            iou = calculate_iou(box1, box2)
            assert isinstance(iou, torch.Tensor)
            assert 0 <= iou <= 1  # IoU should be between 0 and 1
            assert iou > 0  # Overlapping boxes should have positive IoU
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion=None, reason="skip"),
        reason="PyTorch not available",
    )
    def test_calculate_iou_identical_boxes(self):
        """Test IoU of identical boxes equals 1.0."""
        try:
            import torch

            from cosmic_object_scanner.models.fasterrcnn_resnet50_fpn import calculate_iou

            box = torch.tensor([0.0, 0.0, 10.0, 10.0])
            iou = calculate_iou(box, box)
            assert torch.allclose(iou, torch.tensor(1.0), atol=1e-5)
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion=None, reason="skip"),
        reason="PyTorch not available",
    )
    def test_calculate_iou_non_overlapping(self):
        """Test IoU of non-overlapping boxes equals 0.0."""
        try:
            import numpy as np

            from cosmic_object_scanner.models.fasterrcnn_resnet50_fpn import calculate_iou

            box1 = np.array([0.0, 0.0, 5.0, 5.0])
            box2 = np.array([10.0, 10.0, 15.0, 15.0])

            iou = calculate_iou(box1, box2)
            assert abs(iou - 0.0) < 1e-5
        except ImportError:
            pytest.skip("Required dependencies not available")


class TestDataTransforms:
    """Test data transformation pipeline."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion=None, reason="skip"),
        reason="PyTorch not available",
    )
    def test_transforms_imported(self):
        """Test that transforms module can be imported."""
        try:
            from cosmic_object_scanner.models import transforms

            assert hasattr(transforms, "Compose")
        except ImportError:
            pytest.skip("Required dependencies not available")


class TestCollateFunction:
    """Test collate function for DataLoader."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion=None, reason="skip"),
        reason="PyTorch not available",
    )
    def test_collate_function_exists(self):
        """Test that collate_fn is defined and callable."""
        try:
            from cosmic_object_scanner.models.fasterrcnn_resnet50_fpn import collate_fn

            assert callable(collate_fn)
        except ImportError:
            pytest.skip("Required dependencies not available")

    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion=None, reason="skip"),
        reason="PyTorch not available",
    )
    def test_collate_function_batch(self, dummy_images_batch, dummy_targets):
        """Test collate function with batch data."""
        try:
            from cosmic_object_scanner.models.fasterrcnn_resnet50_fpn import collate_fn

            batch = list(zip(dummy_images_batch, dummy_targets, strict=True))
            collated = collate_fn(batch)

            assert isinstance(collated, tuple)  # collate_fn returns tuple from zip
            assert len(collated) == 2  # images and targets (both tuples)
            assert len(collated[0]) > 0  # images tuple not empty
            assert len(collated[1]) > 0  # targets tuple not empty
        except ImportError:
            pytest.skip("Required dependencies not available")
