import os
import torch

from PIL import Image
from pycocotools.coco import COCO


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")

        num_objs = len(anns)
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin, ymin, width, height = anns[i]['bbox']

            # Ensure valid bounding boxes
            if width > 0 and height > 0:
                boxes.append([xmin, ymin, xmin + width, ymin + height])
                labels.append(anns[i]['category_id'])

        # Skip images w/o annotations
        if len(boxes) == 0:
            return self.__getitem__((index + 1) % len(self.ids))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)
