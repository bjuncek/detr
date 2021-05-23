import os
import pandas as pd

import torch
from torchvision.datasets.folder import DatasetFolder, pil_loader

import datasets.transforms as T


class VGGFace2(DatasetFolder):
    def __init__(
        self, root, annotation_path, transforms=None, split="train", datapath=None
    ):
        if split == "val":
            split = "train"
        root = os.path.join(root, split)
        self._transform = transforms
        self.annotations = pd.read_csv(annotation_path)
        super(VGGFace2, self).__init__(
            root, loader=pil_loader, extensions=(".jpg", ".jpeg")
        )

        if datapath is not None:
            self.samples = torch.load(datapath)

        print("Loaded vggface2", self.__len__())

    def __getitem__(self, index):
        p, c = self.samples[index]
        img = pil_loader(p)
        targets = self._get_target(index, p, c, img.size)

        if self._transform is not None:
            img, targets = self._transform(img, targets)

        return img, targets

    def _get_target(self, idx, path, c, imgsize):
        w, h = imgsize

        name_ID = os.path.join(*path.split("/")[-2:]).split(".")[0]
        item = self.annotations[self.annotations.NAME_ID == name_ID]

        boxes = torch.tensor(
            [
                [
                    item.X.item(),
                    item.Y.item(),
                    item.X.item() + item.W.item(),
                    item.Y.item() + item.H.item(),
                ]
            ],
            dtype=torch.float32,
        ).reshape(-1, 4)
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        labels = torch.tensor([c], dtype=torch.int64)
        iscrowd = torch.zeros(boxes.size(0))

        if (boxes[:, 2:] < boxes[:, :2]).all():
            print("shit", boxes)

        return {
            "boxes": boxes,
            "labels": labels,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
            "image_id": torch.tensor([idx]),
        }


def make_default_transforms(image_set, crop=False):
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    # normalize = T.Compose([T.ToTensor()])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        if not crop:
            trans = [
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
        else:
            trans = [
                T.RandomHorizontalFlip(),
                T.RandomResize([256, 324]),
                T.RandomSizeCrop(224, 224),
                normalize,
            ]
        return T.Compose(trans)

    if image_set in ["test", "val"]:
        return T.Compose([T.RandomResize([800], max_size=1333), normalize,])

    raise ValueError(f"unknown {image_set}")


def build(image_set, crop=False):

    ann_set = image_set
    if image_set == "val":
        ann_set = "train"

    datapath = (
        f"/users/korbar/phd/detr_working_copy/datasets/custom_data/vggface2_{image_set}.pth"
        if image_set in ["train", "val"]
        else None
    )

    root = "/work/korbar/datasets/VGGFACE2_raw"
    annotations = f"/work/korbar/datasets/VGGFACE2_raw/bb_landmark/loose_bb_{ann_set}.csv"
    transforms = make_default_transforms(image_set, crop)

    return VGGFace2(
        root,
        annotation_path=annotations,
        split=image_set,
        transforms=transforms,
        datapath=datapath,
    )

