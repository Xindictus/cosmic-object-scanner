import os
from typing import Any

import torch
import torch.utils.data
import torchvision
from PIL.Image import Image
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from cosmic_object_scanner.models import transforms as T  # noqa: N812


def convert_coco_poly_to_mask(segmentations: list[Any], height: int, width: int) -> torch.Tensor:
    masks: list[torch.Tensor] = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks_stacked = torch.stack(masks, dim=0)
    else:
        masks_stacked = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks_stacked


class ConvertCocoPolysToMask:
    def __call__(self, image: Image, target: dict[str, Any]) -> tuple[Image, dict[str, Any]]:
        w, h = image.size

        image_id = target["image_id"]

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes: list[Any] = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes_tensor[:, 2:] += boxes_tensor[:, :2]
        boxes_tensor[:, 0::2].clamp_(min=0, max=w)
        boxes_tensor[:, 1::2].clamp_(min=0, max=h)

        classes: list[Any] = [obj["category_id"] for obj in anno]
        classes_tensor = torch.tensor(classes, dtype=torch.int64)

        segmentations: list[Any] = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints: torch.Tensor | None = None
        if anno and "keypoints" in anno[0]:
            keypoints_list: list[Any] = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints_list, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes_tensor[:, 3] > boxes_tensor[:, 1]) & (boxes_tensor[:, 2] > boxes_tensor[:, 0])
        boxes_tensor = boxes_tensor[keep]
        classes_tensor = classes_tensor[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target_out: dict[str, Any] = {}
        target_out["boxes"] = boxes_tensor
        target_out["labels"] = classes_tensor
        target_out["masks"] = masks
        target_out["image_id"] = image_id
        if keypoints is not None:
            target_out["keypoints"] = keypoints

        # for conversion to coco api
        area_list: list[Any] = [obj["area"] for obj in anno]
        iscrowd_list: list[Any] = [obj["iscrowd"] for obj in anno]
        area_tensor = torch.tensor(area_list)
        iscrowd_tensor = torch.tensor(iscrowd_list)
        target_out["area"] = area_tensor
        target_out["iscrowd"] = iscrowd_tensor

        return image, target_out


def _coco_remove_images_without_annotations(
    dataset: Any, cat_list: list[int] | None = None
) -> torch.utils.data.Subset[Any]:
    def _has_only_empty_bbox(anno: list[Any]) -> bool:
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno: list[Any]) -> int:
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno: list[Any]) -> bool:
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        return _count_visible_keypoints(anno) >= min_keypoints_per_image

    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    result = torch.utils.data.Subset(dataset, ids)
    return result


def convert_to_coco_api(ds: Any) -> COCO:
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset: dict[str, Any] = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"]
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset: Any) -> COCO:
    # Unwrap nested Subsets to reach the underlying CocoDetection dataset.
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


class CocoDetection(torchvision.datasets.CocoDetection):  # type: ignore[misc]
    def __init__(self, img_folder: str, ann_file: str, transforms: Any) -> None:
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx: int) -> tuple[Image, dict[str, Any]]:
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target_dict: dict[str, Any] = {"image_id": image_id, "annotations": target}
        if self._transforms is not None:
            img, target_dict = self._transforms(img, target_dict)
        return img, target_dict


def get_coco(
    root: str,
    image_set: str,
    transforms: Any,
    mode: str = "instances",
    use_v2: bool = False,
    with_masks: bool = False,
) -> CocoDetection:
    anno_file_template = "{}_{}2017.json"
    paths = {
        "train": (
            "train2017",
            os.path.join("annotations", anno_file_template.format(mode, "train")),
        ),
        "val": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val"))),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    img_folder, ann_file = paths[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    if use_v2:
        from torchvision.datasets import wrap_dataset_for_transforms_v2

        dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)
        target_keys = ["boxes", "labels", "image_id"]
        if with_masks:
            target_keys += ["masks"]
        dataset = wrap_dataset_for_transforms_v2(dataset, target_keys=target_keys)
    else:
        # V1 path: wrap transforms with mask conversion.
        t = [ConvertCocoPolysToMask()]
        if transforms is not None:
            t.append(transforms)
        transforms = T.Compose(t)

        dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset  # type: ignore[no-any-return]
