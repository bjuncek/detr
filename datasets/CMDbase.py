import torch
import torchvision.io as IO
import torchvision.transforms.functional as F

import os
import pandas as pd


class CMDBase(object):
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

        return F.to_pil_image(video[0, ...].permute(2, 0, 1))

    def __len__(self) -> int:
        return len(self.clips)

