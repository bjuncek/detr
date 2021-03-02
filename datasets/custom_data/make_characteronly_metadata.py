import pickle, os
import pandas as pd
import torch

annotation_dir = "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/metadata"
failed_path_list = [
        "/users/korbar/phd/detr_working_copy/datasets/custom_data/failed.torchp",
        "/users/korbar/phd/detr_working_copy/datasets/custom_data/new_failed.torchp"]
facetrack_dir = "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/facetracks"
    


# load annotations and make class_to_idx
fd_path = "/scratch/shared/beegfs/maxbain/datasets/CondensedMovies/facetracks/annotations_dict_2.pk"
with open(fd_path, "rb",) as f:
    face_dets = pickle.load(f)
    
ACTORS = set()
for key, value in face_dets.items():
    for ft in value:
        if ft is not None:
            ACTORS.add(ft[0])
lACTORS = sorted(list(ACTORS))
class_to_idx = {lACTORS[i]: i for i in range(len(lACTORS))}
                 

# load clips and such

for split in ["train", "val"]:
    print(split)
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

    # now go throught everything and save files to a predefined folder 

    for df_id in range(len(clips)):
        if df_id % 200 == 0:
            print(df_id)
        clip_id = clips.iloc[df_id]['videoid']
        video_year = clips.iloc[df_id]['upload_year']
        if clip_id + ".mkv" in face_dets:
            value = face_dets[clip_id + ".mkv"]
        else:
            missing.append(clip_id)
            continue
        # get identified values in the facetrack
        known_actors_in_clip = [i for i in range(len(value)) if value[i] is not None]
        # load facetrack and database for the clip
        fd_path = os.path.join(facetrack_dir, str(int(video_year)), clip_id + ".mkvface_dets.pk")
        db_path = os.path.join(facetrack_dir, str(int(video_year)), clip_id + ".mkvdatabase.pk")
        with open(fd_path, "rb",) as f:
                detections = pickle.load(f)
        with open(db_path, "rb",) as f:
                database = pickle.load(f)


        # here we have all annotations for the named files
        annotations = [value[i] for i in known_actors_in_clip]
        # all the facetracks for the named characters
        track_ranges = [database['index_into_facedetfile'][i] for i in known_actors_in_clip]
        tracks = [detections[track] for track in track_ranges]
        # all the features for the named characters
        feats = [database['feats'][i] for i in known_actors_in_clip]

        # make sure all correspond correctly
        assert len(annotations) == len(tracks) == len(feats)

        # then create new type of annotation
        new_ = {"actor":[], "class":[], "class_confidence":[], "facetrack":[], "facetrack_feature":[]}
        for t in range(len(annotations)):
            new_["actor"].append(annotations[t][0])
            new_["class"].append(class_to_idx[annotations[t][0]])
            new_["class_confidence"].append(annotations[t][1])
            new_["facetrack"].append(tracks[t])
            new_["facetrack_feature"].append(feats[t])

        torch.save(new_, f"/work/korbar/CMD/named_metadata/{clip_id}.th")