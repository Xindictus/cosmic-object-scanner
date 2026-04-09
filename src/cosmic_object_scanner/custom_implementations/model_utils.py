import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from cosmic_object_scanner.custom_implementations.constants import ANCHORS, device
from cosmic_object_scanner.custom_implementations.utils import (
    convert_cells_to_bboxes,
    iou,
    keep_prominent_boxes,
    nms,
)


class YOLOLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, anchors: torch.Tensor
    ) -> torch.Tensor:
        # Identifying which cells in target have objects
        # and which have no objects
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # Calculating No object loss
        no_object_loss = self.bce(
            (pred[..., 0:1][no_obj]),
            (target[..., 0:1][no_obj]),
        )

        # Reshaping anchors to match predictions
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        # Box prediction confidence
        box_preds = torch.cat(
            [self.sigmoid(pred[..., 1:3]), torch.exp(pred[..., 3:5]) * anchors], dim=-1
        )
        # Calculating intersection over union for prediction and target
        ious = iou(box_preds[obj], target[..., 1:5][obj]).detach()
        # Calculating Object loss
        object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # Predicted box coordinates
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3])
        # Target box coordinates
        target[..., 3:5] = torch.log(1e-6 + target[..., 3:5] / anchors)
        # Calculating box coordinate loss
        box_loss = self.mse(pred[..., 1:5][obj], target[..., 1:5][obj])

        # Claculating class loss
        class_loss = self.cross_entropy((pred[..., 5:][obj]), target[..., 5][obj].long())

        # Total loss
        total_loss: torch.Tensor = box_loss + object_loss + no_object_loss + class_loss
        return total_loss


def training_loop(
    loader: Any,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    scaled_anchors: Any,
) -> float:
    # Creating a progress bar
    progress_bar = tqdm(loader, leave=True)

    # Initializing a list to store the losses
    losses = []

    # Iterating over the training data
    for _, (x, y) in enumerate(progress_bar):
        x = x.to(device)
        y0, y1, y2 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )

        with torch.cuda.amp.autocast():
            # Getting the model predictions
            outputs = model(x)
            # Calculating the loss at each scale
            loss = (
                loss_fn(outputs[0], y0, scaled_anchors[0])
                + loss_fn(outputs[1], y1, scaled_anchors[1])
                + loss_fn(outputs[2], y2, scaled_anchors[2])
            )

        # Add the loss to the list
        losses.append(loss.item())

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagate the loss
        scaler.scale(loss).backward()

        # Optimization step
        scaler.step(optimizer)

        # Update the scaler for next iteration
        scaler.update()

        # update progress bar with loss
        mean_loss = sum(losses) / len(losses)
        progress_bar.set_postfix(loss=mean_loss)

    return float(mean_loss)


def evaluate(loader: Any, model: nn.Module, loss_fn: nn.Module, scaled_anchors: Any) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y0, y1, y2 = y[0].to(device), y[1].to(device), y[2].to(device)
            outputs = model(x)
            loss = (
                loss_fn(outputs[0], y0, scaled_anchors[0])
                + loss_fn(outputs[1], y1, scaled_anchors[1])
                + loss_fn(outputs[2], y2, scaled_anchors[2])
            )
            losses.append(loss.item())

    model.train()
    filtered_losses = [x for x in losses if not math.isnan(x)]
    return float(sum(filtered_losses) / len(filtered_losses))


def test_model(
    test_loader: Any, model: nn.Module, device: torch.device
) -> tuple[list[Any], list[Any]]:
    y_true = []
    y_pred = []
    # Getting a sample image from the test data loader
    for x, _y, labels in iter(test_loader):
        x = x.to(device)

        model.eval()
        with torch.no_grad():
            # Getting the model predictions
            output = model(x)
            # Getting the bounding boxes from the predictions
            bboxes: list[list[Any]] = [[] for _ in range(x.shape[0])]

            # Getting bounding boxes for each scale
            for i in range(3):
                batch_size, num_anchors, s, _, _ = output[i].shape
                anchors = (
                    torch.tensor(ANCHORS)
                    * torch.tensor(s).unsqueeze(0).unsqueeze(0).repeat(1, 3, 2)
                ).to(device)
                anchor = anchors[i]
                boxes_scale_i = convert_cells_to_bboxes(output[i], anchor, s=s, is_predictions=True)
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box

        # Plotting the image with bounding boxes for each image in the batch
        # Applying non-max suppression to remove overlapping bounding boxes
        nms_boxes = nms(bboxes[0], iou_threshold=0.9, threshold=0.8)
        # Plotting the image with bounding boxes
        final_preds_list = keep_prominent_boxes(nms_boxes, 0.5, 1)

        final_preds_array: Any = np.array(final_preds_list)
        if final_preds_array.size == 0:
            # Handle empty array case
            final_preds = []
        elif final_preds_array.ndim == 2 and final_preds_array.shape[1] > 0:
            # Extract the first element of each sublist and convert to integers
            final_preds = final_preds_array[:, 0].astype(int).tolist()
        else:
            # Handle case where array might not be 2D or might not have sublists
            final_preds = []

        labels_len = len(labels)
        preds_len = len(final_preds)

        for i in range(labels_len):
            labels[i] = int(labels[i][0])

        if labels_len == 0:
            labels = [-1]
            labels_len = 1

        if labels_len < preds_len:
            final_preds = final_preds[:labels_len]
        elif labels_len > preds_len:
            ext_len = labels_len - preds_len
            final_preds.extend([-1] * ext_len)

        y_true.extend(labels)
        y_pred.extend(final_preds)

        if len(y_true) != len(y_pred):
            print("ERROR!")
            break

    return y_true, y_pred
