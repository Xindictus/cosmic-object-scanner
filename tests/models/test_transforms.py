"""Unit tests for models.transforms module."""

import pytest
import torch

from cosmic_object_scanner.models.transforms import (
    Compose,
    PILToTensor,
    RandomHorizontalFlip,
    ToDtype,
)


@pytest.mark.unit
class TestCompose:
    """Tests for the Compose transform."""

    def test_identity_composition(self) -> None:
        def identity(
            img: torch.Tensor, target: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            return img, target

        compose = Compose([identity, identity])
        img = torch.randn(3, 32, 32)
        target: dict[str, torch.Tensor] = {"boxes": torch.tensor([[0, 0, 10, 10]])}
        out_img, out_target = compose(img, target)
        assert torch.equal(out_img, img)
        assert torch.equal(out_target["boxes"], target["boxes"])

    def test_empty_composition(self) -> None:
        compose = Compose([])
        img = torch.randn(3, 32, 32)
        target: dict[str, torch.Tensor] = {}
        out_img, _ = compose(img, target)
        assert torch.equal(out_img, img)


@pytest.mark.unit
class TestRandomHorizontalFlip:
    """Tests for the RandomHorizontalFlip transform."""

    def test_no_flip_p0(self) -> None:
        transform = RandomHorizontalFlip(p=0.0)
        img = torch.randn(3, 32, 32)
        target: dict[str, torch.Tensor] = {
            "boxes": torch.tensor([[5.0, 5.0, 15.0, 15.0]]),
        }
        out_img, out_target = transform(img, target)
        assert torch.equal(out_img, img)
        assert out_target is not None
        assert torch.equal(out_target["boxes"], target["boxes"])

    def test_always_flip_p1(self) -> None:
        transform = RandomHorizontalFlip(p=1.0)
        img = torch.randn(3, 32, 32)
        target: dict[str, torch.Tensor] = {
            "boxes": torch.tensor([[5.0, 5.0, 15.0, 15.0]]),
        }
        out_img, out_target = transform(img, target)
        # Image should be flipped
        assert not torch.equal(out_img, img)
        assert out_target is not None
        # Boxes should be horizontally flipped
        assert out_target["boxes"].shape == (1, 4)

    def test_with_none_target(self) -> None:
        transform = RandomHorizontalFlip(p=1.0)
        img = torch.randn(3, 32, 32)
        out_img, out_target = transform(img, None)
        assert out_target is None


@pytest.mark.unit
class TestToDtype:
    """Tests for the ToDtype transform."""

    def test_dtype_conversion(self) -> None:
        transform = ToDtype(torch.float32, scale=False)
        img = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
        out_img, _ = transform(img, None)
        assert out_img.dtype == torch.float32

    def test_dtype_with_scale(self) -> None:
        transform = ToDtype(torch.float32, scale=True)
        img = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
        out_img, _ = transform(img, None)
        assert out_img.dtype == torch.float32
        assert out_img.max() <= 1.0


@pytest.mark.unit
class TestPILToTensor:
    """Tests for the PILToTensor transform."""

    def test_pil_conversion(self) -> None:
        from PIL import Image

        transform = PILToTensor()
        img = Image.new("RGB", (32, 32), color="red")
        out_img, _ = transform(img, None)
        assert isinstance(out_img, torch.Tensor)
        assert out_img.shape == (3, 32, 32)
