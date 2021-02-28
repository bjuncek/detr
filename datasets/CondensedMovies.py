import os, pickle, random
import pandas as pd
import pdb
from tqdm import tqdm
import torch
import decord
from decord import VideoReader, cpu
import torchvision.io as IO
import torchvision.transforms.functional as F


# for debugging purposes atm
import datasets.transforms as T


class CondensedMoviesBase(object):
    def __init__(
        self,
        data_dir: str,
        facetrack_dir: str,
        annotation_dir: str,
        transforms=None,
        split="train",
        failed_path=None,
    ):
        self.data_dir = data_dir
        self.facetrack_dir = facetrack_dir
        self._load_clips(annotation_dir, split, failed_path)
        self._transforms = transforms

    def _load_clips(self, annotation_dir, split, failed_path_list):
        clips = pd.read_csv(f"{annotation_dir}/clips.csv")
        split_data = pd.read_csv(f"{annotation_dir}/split.csv").set_index("imdbid")
        ids = split_data[split_data["split"] == split].index
        f = clips["imdbid"].isin(ids)
        clips = clips[f]
        clips = clips[clips["year"] <= 2020]
        clips = clips.reset_index(drop=True)
        if failed_path_list is not None:
            for failed_path in failed_path_list:
                failed_ids = torch.load(failed_path)[split]
                bad_df = clips.index.isin(failed_ids)
                clips = clips[~bad_df]
                clips = clips.reset_index(drop=True)
        self.clips = clips

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

    def _get_video_path(self, video_year: str, video_ID: str) -> str:
        p = os.path.join(self.data_dir, video_year, video_ID + ".mkv")
        if not os.path.isfile(p):
            # pdb.set_trace()
            raise Exception(f"path to video {p} does not exist")
        return p

    def _get_video_frame(self, video_year, video_ID, frame_id):
        p = self._get_video_path(video_year, video_ID)
        # videos are all at 25 fps, so to get rought seconds cound we divide that
        s_entry = (frame_id - 1) / 25
        video, _, _ = IO.read_video(p, s_entry, s_entry + 0.5, pts_unit="sec")
        # vr = VideoReader(p, ctx=cpu(0))
        # if frame_id > len(vr):
        #     frame_id = len(vr) - 1
        # frame = vr[frame_id - 1]  # this should be a tensor

        return F.to_pil_image(video[0, ...].permute(2, 0, 1))

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
            labels.append(det[5])
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

    def __len__(self) -> int:
        return len(self.clips)


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


def build(image_set):
    decord.bridge.set_bridge("torch")
    data_dir = (
        "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/videos_with_border"
    )
    facetracks = "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/facetracks"
    annotations = "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/metadata"
    failed = [
        "/users/korbar/phd/detr_working_copy/failed.pth",
        "/users/korbar/phd/detr_working_copy/new_failed.pth",
    ]

    return CondensedMoviesBase(
        data_dir,
        facetracks,
        annotations,
        transforms=make_default_transforms(image_set),
        split=image_set,
        failed_path=failed,
    )
