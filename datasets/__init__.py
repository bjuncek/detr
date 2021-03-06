# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .CondensedMovies import build_detection as build_cmd_det
from .CondensedMovies import build_character as calvin
from .wider import build
from .vggface2 import build as buildvgg
from .mot_data import build_MOT


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == "coco":
        return build_coco(image_set, args)
    if args.dataset_file == "cmdd":
        return build_cmd_det(image_set)
    if args.dataset_file == "cmdc":
        return calvin(image_set)
    if args.dataset_file == "wider":
        return build(image_set, args)
    if args.dataset_file == "vggface2":
        return buildvgg(image_set, crop=args.crop)
    if args.dataset_file == "MOT17":
        return build_MOT(image_set)
    if args.dataset_file == "coco_panoptic":
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f"dataset {args.dataset_file} not supported")
