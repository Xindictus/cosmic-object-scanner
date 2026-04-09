"""Unit tests for models.utils (MetricLogger, SmoothedValue, collate_fn, etc)."""

import pytest
import torch

from cosmic_object_scanner.models.utils import (
    MetricLogger,
    SmoothedValue,
    collate_fn,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    mkdir,
    reduce_dict,
)


@pytest.mark.unit
class TestSmoothedValue:
    """Tests for SmoothedValue tracker."""

    def test_initial_state(self) -> None:
        sv = SmoothedValue()
        assert sv.count == 0
        assert sv.total == 0.0

    def test_update(self) -> None:
        sv = SmoothedValue(window_size=5)
        sv.update(10.0)
        sv.update(20.0)
        assert sv.count == 2
        assert sv.total == 30.0

    def test_median(self) -> None:
        sv = SmoothedValue(window_size=5)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            sv.update(v)
        assert sv.median == pytest.approx(3.0)

    def test_avg(self) -> None:
        sv = SmoothedValue(window_size=10)
        for v in [2.0, 4.0, 6.0]:
            sv.update(v)
        assert sv.avg == pytest.approx(4.0)

    def test_global_avg(self) -> None:
        sv = SmoothedValue(window_size=2)
        for v in [10.0, 20.0, 30.0]:
            sv.update(v)
        assert sv.global_avg == pytest.approx(20.0)

    def test_max(self) -> None:
        sv = SmoothedValue(window_size=5)
        for v in [1.0, 5.0, 3.0]:
            sv.update(v)
        assert sv.max == pytest.approx(5.0)

    def test_value(self) -> None:
        sv = SmoothedValue(window_size=5)
        sv.update(7.0)
        sv.update(9.0)
        assert sv.value == pytest.approx(9.0)

    def test_str(self) -> None:
        sv = SmoothedValue(window_size=5)
        sv.update(5.0)
        s = str(sv)
        assert isinstance(s, str)
        assert len(s) > 0


@pytest.mark.unit
class TestMetricLogger:
    """Tests for MetricLogger."""

    def test_update_and_str(self) -> None:
        logger = MetricLogger()
        logger.update(loss=0.5, acc=0.9)
        s = str(logger)
        assert "loss" in s
        assert "acc" in s

    def test_update_with_tensor(self) -> None:
        logger = MetricLogger()
        logger.update(loss=torch.tensor(0.5))
        assert logger.loss.count == 1

    def test_getattr_missing(self) -> None:
        logger = MetricLogger()
        with pytest.raises(AttributeError):
            _ = logger.nonexistent_meter

    def test_add_meter(self) -> None:
        logger = MetricLogger()
        sv = SmoothedValue(window_size=10)
        logger.add_meter("custom", sv)
        assert "custom" in logger.meters

    def test_log_every(self) -> None:
        logger = MetricLogger()
        items = list(range(5))
        results = list(logger.log_every(items, print_freq=2, header="Test"))
        assert results == items


@pytest.mark.unit
class TestCollateFn:
    """Tests for the collate function."""

    def test_collate_tuples(self) -> None:
        batch = [(1, "a"), (2, "b"), (3, "c")]
        result = collate_fn(batch)
        assert result == ((1, 2, 3), ("a", "b", "c"))


@pytest.mark.unit
class TestDistributedUtilities:
    """Tests for distributed training utilities (single-process fallbacks)."""

    def test_is_dist_not_initialized(self) -> None:
        # In test environment, distributed is not initialized
        result = is_dist_avail_and_initialized()
        assert result is False

    def test_get_world_size_single(self) -> None:
        assert get_world_size() == 1

    def test_get_rank_zero(self) -> None:
        assert get_rank() == 0

    def test_is_main_process(self) -> None:
        assert is_main_process() is True

    def test_reduce_dict_single_process(self) -> None:
        d = {"loss": torch.tensor(0.5), "acc": torch.tensor(0.9)}
        result = reduce_dict(d)
        assert result is d  # single-process returns input unchanged

    def test_mkdir(self, tmp_path: pytest.TempPathFactory) -> None:
        new_dir = str(tmp_path) + "/test_subdir"  # type: ignore[operator]
        mkdir(new_dir)
        import os

        assert os.path.isdir(new_dir)

    def test_mkdir_existing(self, tmp_path: pytest.TempPathFactory) -> None:
        existing = str(tmp_path)
        # Should not raise for existing directory
        mkdir(existing)
