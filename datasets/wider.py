import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

import datasets.transforms as T


class WIDER(object):
    def __init__(self, dataset_root, split="train", transforms=None, mode="invalid"):
        self.dataset_root = Path(dataset_root)
        self.mode = mode
        self._transform = transforms

        self.data = []
        self.split = split

        self.load(dataset_root)
        print("Dataset loaded")
        print("{0} samples in the {1} dataset".format(len(self.data), self.split))

    def load(self, path):
        """Load the dataset from the text file."""

        path = Path(
            os.path.join(
                path, "wider_face_split", f"wider_face_{self.split}_bbx_gt.txt"
            )
        )
        lines = open(path).readlines()
        self.data = []
        idx = 0

        while idx < len(lines):
            img = lines[idx].strip()
            idx += 1
            n = int(lines[idx].strip())
            idx += 1

            bboxes = np.empty((n, 10))

            if n == 0:
                idx += 1
            else:
                for b in range(n):
                    bboxes[b, :] = [abs(float(x)) for x in lines[idx].strip().split()]
                    idx += 1

            # remove invalid bboxes where w or h are 0
            invalid = np.where(np.logical_or(bboxes[:, 2] == 0, bboxes[:, 3] == 0))
            bboxes = np.delete(bboxes, invalid, 0)

            # bounding boxes are 1 indexed so we keep them like that
            # and treat them as abstract geometrical objects
            # We only need to worry about the box indexing when actually rendering them

            # convert from (x, y, w, h) to (x1, y1, x2, y2)
            # We work with the two point representation
            # since cropping becomes easier to deal with
            # -1 to ensure the same representation as in Matlab.
            bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] - 1
            bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] - 1

            datum = {
                "img_path": img,
                "bboxes": bboxes[:, 0:4],
                "blur": bboxes[:, 4],
                "expression": bboxes[:, 5],
                "illumination": bboxes[:, 6],
                "invalid": bboxes[:, 7],
                "occlusion": bboxes[:, 8],
                "pose": bboxes[:, 9],
            }

            self.data.append(datum)

    def __len__(self):
        return len(self.data)

    def _get_target(self, idx, datum, imgsize, mode="invalid"):

        w, h = imgsize
        boxes = torch.as_tensor(datum["bboxes"], dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        iscrowd = torch.zeros(boxes.size(0))

        labels = torch.tensor(datum[mode], dtype=torch.int64)

        return {
            "boxes": boxes,
            "labels": labels,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
            "image_id": torch.tensor([idx]),
        }

    def __getitem__(self, index):
        datum = self.data[index]

        image_root = self.dataset_root / "WIDER_{0}".format(self.split)
        image_path = image_root / "images" / datum["img_path"]
        img = Image.open(image_path).convert("RGB")
        targets = self._get_target(index, datum, img.size, self.mode)

        if self._transform is not None:
            img, targets = self._transform(img, targets)

        return img, targets


def make_default_transforms(image_set):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    # normalize = T.Compose([T.ToTensor()])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose([T.RandomResize([800], max_size=1333), normalize,])

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):

    assert args.wider_mode in [
        "blur",
        "expression",
        "illumination",
        "occlusion",
        "pose",
        "invalid",
    ]
    root = "/work/korbar/WIDER"
    transforms = make_default_transforms(image_set)

    return WIDER(root, split=image_set, transforms=transforms, mode=args.wider_mode)
