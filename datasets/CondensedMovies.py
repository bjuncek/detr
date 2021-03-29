import os
import pickle
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from datasets.CMDbase import CMDBase
import datasets.transforms as T


class CondensedMoviesCharacter(CMDBase):
    def __init__(
        self,
        data_dir: str,
        facetrack_dir: str,
        annotation_dir: str,
        transforms=None,
        split="train",
        failed_path=None,
    ):
        super(CondensedMoviesCharacter, self).__init__(
            data_dir, facetrack_dir, annotation_dir, transforms, split, failed_path
        )

    def __getitem__(self, index: int):
        # get the info from the database
        target_clip = self.clips.iloc[index]
        video_year = str(int(target_clip["upload_year"]))
        video_ID = str(target_clip["videoid"])

        df = torch.load(os.path.join(self.facetrack_dir, video_ID + ".th"))
        frame_num = self._get_frame_num(df)
        img = self._get_video_frame(video_year, video_ID, frame_num)
        target = self._get_target(df, frame_num, img.size, index)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def _get_frame_num(self, df):
        track_idx = random.randrange(len(df["actor"]))
        frame_num = random.choice(df["facetrack_frame"][track_idx])
        return frame_num

    def _get_target(self, df, frame_num, imgsize, idx):
        # check if there is another track in there
        # returning track_idx, frame_idx
        idxs = []
        for i in range(len(df["facetrack_frame"])):
            frame_id = np.where(df["facetrack_frame"][i] == frame_num)[0]
            if len(frame_id) != 0:
                idxs.append((i, frame_id[0]))

        boxes, labels, areas, iscrowd, confidence, facetracks = ([], [], [], [], [], [])
        # LOOP OVER THE DETECTIONS AND ADD THEM TO THE TARGET
        for det in idxs:
            track_id, frame_id = det
            labels.append(df["class"][track_id])
            xyhw = df["facetrack_xyhw"][track_id][frame_id]
            boxes.append(xyhw)
            areas.append(xyhw[2] * xyhw[3])
            iscrowd.append(0)
            confidence.append(df["class_confidence"][track_id])
            facetracks.append(df["facetrack_feature"][track_id])

        w, h = imgsize
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        labels = torch.tensor(labels, dtype=torch.int64)
        confidence = torch.tensor(confidence)
        iscrowd = torch.tensor(iscrowd)
        image_id = torch.tensor([frame_num])
        clips_id = torch.tensor([idx])
        embeddings = torch.tensor(facetracks)

        return {
            "boxes": boxes,
            "labels": labels,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
            "class_confidence": confidence,
            "image_id": image_id,
            "clips_idx": clips_id,
            "embeddings": embeddings,
        }


class CondensedMoviesDetection(CMDBase):
    def __init__(
        self,
        data_dir: str,
        facetrack_dir: str,
        annotation_dir: str,
        transforms=None,
        split="train",
        failed_path=None,
    ):
        super(CondensedMoviesDetection, self).__init__(
            data_dir, facetrack_dir, annotation_dir, transforms, split, failed_path
        )

    def __getitem__(self, index: int):
        # get the info from the database
        target_clip = self.clips.iloc[index]
        video_year = str(int(target_clip["upload_year"]))
        video_ID = str(target_clip["videoid"])

        # load the database and randomly select frame_id
        face_dets, database = self._load_facetracks(video_year, video_ID)
        frame_id, detections = self._get_frame_id(face_dets, database)
        vm = {"frame_id": frame_id, "upload_year": video_year, "videoid": video_ID}

        # actually load video frame and generate the target dict
        img = self._get_video_frame(video_year, video_ID, frame_id)
        target = self._get_target_dict(detections, img.size, vm, index)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target

    def _load_facetracks(self, video_year, video_ID):
        fd_path = os.path.join(
            self.facetrack_dir, video_year, video_ID + ".mkvface_dets.pk"
        )
        db_path = os.path.join(
            self.facetrack_dir, video_year, video_ID + ".mkvdatabase.pk"
        )
        if not os.path.isfile(fd_path):
            # pdb.set_trace()
            raise Exception(f"path to face det {fd_path} does not exist")
        if not os.path.isfile(db_path):
            # pdb.set_trace()
            raise Exception(f"path to face database {db_path} does not exist")
        with open(fd_path, "rb",) as f:
            face_dets = pickle.load(f)

        with open(db_path, "rb",) as f:
            database = pickle.load(f)
        return face_dets, database

    def _get_frame_id(self, face_dets, database, _frame_id=None):
        # SELECT TRACK AT RANDOM

        frames = random.choice(database["index_into_facedetfile"])
        # GET A FRAME FROM THAT DETECTION AT RANDOM
        # FIXME: AT THE MOMENT TAKE FIRST
        # frame_id = int(face_dets[random.sample(frames, 1)[0]][0])
        frame_id = int(face_dets[frames[0]][0])

        # DEBUGGING:
        if _frame_id:
            frame_id = _frame_id
        # FIND ALL OTHER DETECTIONS FOR THAT FRAME
        det_ids = face_dets[face_dets[:, 0] == frame_id]
        return frame_id, det_ids

    def _get_target_dict(self, det_ids, imgsize, vm, idx):
        boxes, labels, areas, iscrowd, confidence = (
            [],
            [],
            [],
            [],
            [],
        )
        # LOOP OVER THE DETECTIONS AND ADD THEM TO THE TARGET
        for det in det_ids:
            boxes.append(
                [int(det[1]), int(det[2]), int(det[3]), int(det[4])]
            )  # [x1, y1, x2, y2]
            areas.append(det[3] * det[4])
            labels.append(1)  # labels are 0 if face, 1 otherwise
            iscrowd.append(0)
            confidence.append(det[6])

        w, h = imgsize

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        labels = torch.tensor(labels, dtype=torch.int64)
        confidence = torch.tensor(confidence)

        iscrowd = torch.tensor(iscrowd)
        image_id = torch.tensor([idx])

        return {
            "boxes": boxes,
            "labels": labels,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
            "detection_confidence": confidence,
            "image_id": image_id,
        }


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


def build_detection(image_set):
    data_dir = (
        "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/videos_with_border"
    )
    facetracks = "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/facetracks"
    annotations = "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/metadata"
    failed = [
        "/users/korbar/phd/detr_working_copy/datasets/custom_data/failed.torchp",
        "/users/korbar/phd/detr_working_copy/datasets/custom_data/new_failed.torchp",
    ]

    return CondensedMoviesDetection(
        data_dir,
        facetracks,
        annotations,
        transforms=make_default_transforms(image_set),
        split=image_set,
        failed_path=failed,
    )


def build_character(image_set):
    data_dir = (
        "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/videos_with_border"
    )
    facetracks = "/work/korbar/CMD/named_metadata"
    annotations = "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/metadata"
    failed = [
        "/users/korbar/phd/detr_working_copy/datasets/custom_data/failed.torchp",
        "/users/korbar/phd/detr_working_copy/datasets/custom_data/new_failed.torchp",
        "/users/korbar/phd/detr_working_copy/datasets/custom_data/char_failed.torchp",
    ]

    return CondensedMoviesCharacter(
        data_dir,
        facetracks,
        annotations,
        transforms=make_default_transforms(image_set),
        split=image_set,
        failed_path=failed,
    )
