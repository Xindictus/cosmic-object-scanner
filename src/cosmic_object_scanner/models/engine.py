import math
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.detection.mask_rcnn

from cosmic_object_scanner.models import utils
from cosmic_object_scanner.models.coco_eval import CocoEvaluator
from cosmic_object_scanner.models.coco_utils import get_coco_api_from_dataset


def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    data_loader: Any,
    device: torch.device,
    epoch: int,
    print_freq: int,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Any:
    """Train model for one epoch.

    Args:
        model: The neural network model to train.
        optimizer: The optimizer to use for training.
        data_loader: DataLoader providing training batches.
        device: Device to run training on (CPU or GPU).
        epoch: Current epoch number.
        print_freq: Frequency of printing metrics (every N batches).
        scaler: Optional gradient scaler for mixed precision training.

    Returns:
        MetricLogger object containing training metrics and statistics.

    Raises:
        SystemExit: If loss becomes NaN or infinite.
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model: nn.Module) -> list[str]:
    """Determine IoU types supported by the model.

    Args:
        model: The detection model to analyze.

    Returns:
        List of supported IoU types (e.g., ['bbox'], ['bbox', 'segm']).
    """
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model: nn.Module, data_loader: Any, device: torch.device) -> CocoEvaluator:
    """Evaluate model on a test dataset using COCO metrics.

    Args:
        model: The detection model to evaluate.
        data_loader: DataLoader providing test batches.
        device: Device to run evaluation on (CPU or GPU).

    Returns:
        CocoEvaluator object with computed metrics.

    Note:
        Sets model to eval mode and runs with inference_mode for efficiency.
    """
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
