import argparse
import os
import time
from os import path as osp


import motmetrics as mm
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from tracking.dsets.factory import Datasets
from tracking.fpn_test import FRCNN_FPN
from models.detracker import build
from tracking.oracle import OracleTracker
from tracking.reid.resnet import ReIDNetwork_resnet50
from tracking.tracker import Tracker
from tracking.utils import (evaluate_mot_accums, get_mot_accum,
                            interpolate_tracks, plot_sequence)
from itertools import cycle


mm.lap.default_solver = 'lap'


# ex = Experiment()

# ex.add_config('experiments/cfgs/tracktor.yaml')
# ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')

def get_output_dir(outdir, name=""):
  outdir = osp.abspath(osp.join(outdir, name, 'tracktor'))
  #if weights_filename is None:
  #  weights_filename = 'default'
  #outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def add_reid_config(reid_models, obj_detect_models, dataset):
    if isinstance(reid_models, str):
        reid_models = [reid_models, ] * len(dataset)

    # if multiple reid models are provided each is applied
    # to a different dataset
    if len(reid_models) > 1:
        assert len(dataset) == len(reid_models)

    if isinstance(obj_detect_models, str):
        obj_detect_models = [obj_detect_models, ] * len(dataset)
    if len(obj_detect_models) > 1:
        assert len(dataset) == len(obj_detect_models)


def main(args, dataset, load_results, frame_range, interpolate, write_images, oracle=None):
    

    output_dir = osp.join(get_output_dir(args.out_dir, args.name))

    if not osp.exists(output_dir):
        os.makedirs(output_dir)


    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    print("Initializing object detector(s).")

    obj_detects = []
    #FIXME: Bruno fix this to reflect DETRa
    if not args.use_detr:
        obj_detect = FRCNN_FPN(num_classes=2)
        obj_detect.load_state_dict(torch.load(args.obj_detect_model,
                                map_location="cpu"))
    else:
        obj_detect = build(args)
    
    obj_detect.eval()
    if torch.cuda.is_available():
        obj_detect.cuda()
    obj_detects.append(obj_detect)

    # reid
    print("Initializing reID network(s).")
    reid_networks = []
    reid_cfg = os.path.join(os.path.dirname(args.reid_model), 'sacred_config.yaml')
    reid_cfg = yaml.safe_load(open(reid_cfg))

    reid_network = ReIDNetwork_resnet50(pretrained=False, **reid_cfg['model_args'])
    reid_network.load_state_dict(torch.load(args.reid_model, map_location="cpu"))
    reid_network.eval()
    if torch.cuda.is_available():
        reid_network.cuda()

    reid_networks.append(reid_network)

    #FIXME: where is tracker coming from
    # tracktor
    # if oracle is not None:
    #     tracker = OracleTracker(
    #         obj_detect, reid_network, tracker, oracle)
    # else:
    tracker = Tracker(obj_detect, reid_network, vars(args))

    if args.profile:
        import torch.autograd.profiler as profiler


    time_total = 0
    num_frames = 0
    mot_accums = []
    dset_file = dataset
    dataset = Datasets(dataset)

    print(dataset)
    for seq, obj_detect, reid_network in zip(dataset, cycle(obj_detects), cycle(reid_networks)):
        print("IN da loop once")
        obj_detect.eval()
        tracker.obj_detect = obj_detect
        tracker.reid_network = reid_network
        tracker.reset()

        print(f"Tracking: {seq}")

        start_frame = int(frame_range['start'] * len(seq))
        end_frame = int(frame_range['end'] * len(seq))

        seq_loader = DataLoader(torch.utils.data.Subset(seq, range(start_frame, end_frame)))
        num_frames += len(seq_loader)

        results = {}
        if load_results:
            print("Reading results")
            results = seq.load_results(output_dir)
        if not results:
            start = time.time()

            with torch.no_grad():
                for frame_data in tqdm(seq_loader):
                    tracker.step(frame_data)

            results = tracker.get_results()

            time_total += time.time() - start

            print(f"Tracks found: {len(results)}")
            print(f"Runtime for {seq}: {time.time() - start :.2f} s.")

            if interpolate:
                results = interpolate_tracks(results)

            print(f"Writing predictions to: {output_dir}")
            seq.write_results(results, output_dir)

        if seq.no_gt:
            print("No GT data for evaluation available.")
        else:
            mot_accums.append(get_mot_accum(results, seq_loader))

        if write_images:
            plot_sequence(
                results,
                seq,
                osp.join(output_dir, str(dataset), str(seq)),
                write_images)

    if time_total:
        print(f"Tracking runtime for all sequences (without evaluation or image writing): "
                f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
    if mot_accums:
        print("Evaluation:")
        print(mot_accums)
        evaluate_mot_accums(mot_accums, 
                            [str(s) for s in dataset if not s.no_gt],
                            generate_overall=True)

def parse_args():
    parser = argparse.ArgumentParser("Set transformer detector and tracker", add_help=False)
    
    parser.add_argument("--out_dir", default="/work/korbar/TRACKING_experiments", type=str)
    parser.add_argument("--name", default="test", type=str)
    # Tracking params
    parser.add_argument("--detection_person_thresh", default=0.5, type=float)
    parser.add_argument("--regression_person_thresh", default=0.5, type=float)
    parser.add_argument("--detection_nms_thresh", default=0.5, type=float)
    parser.add_argument("--regression_nms_thresh", default=0.5, type=float)


    parser.add_argument( "--skip_reid",dest='do_reid', action="store_false", default=True)
    parser.add_argument("--reid_sim_threshold", default=2.0, type=float)
    parser.add_argument("--reid_iou_threshold", default=0.2, type=float)

    parser.add_argument("--num_classes", default=1, type=int)

    parser.add_argument(
        "--pooling_dim",
        default=0,
        type=int,
        help="Type of pooling to apply to the featuremap",
    )
    parser.add_argument(
        "--pooling_method",
        default="none",
        type=str,
        choices=("none", "avgpool", "transformer_pool", "avghack", "encoder_pool"),
        help="Type of pooling to apply to the featuremap",
    )
    parser.add_argument(
        "--obj_detect_model", type=str, default="/work/korbar/datasets/MOT/output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model"
    )
    parser.add_argument(
        "--detr_detect_model", type=str, default="/work/korbar/DETR_experiments/MOT_fromB2_BBOX2/306471/checkpoint.pth"
    )
    parser.add_argument(
        "--reid_model", type=str, default="/work/korbar/datasets/MOT/output/tracktor/reid/res50-mot17-batch_hard/ResNet_iter_25245.pth"
    )

    parser.add_argument( "--use_detr", action="store_true", default=False)
    parser.add_argument(
        "--add_class", default=False, action="store_true",
    )
    parser.add_argument(
        "--profile", default=False, action="store_true",
    )

    args = parser.parse_args()

    ###
    ### default DETR args
    ###
    args.dataset_file = "MOT17"
    args.device = "cuda"
    # backbone
    args.lr_backbone = 0
    args.masks = False
    args.backbone = "resnet50"
    # transformer
    args.hidden_dim = 256
    args.dropout = 0.1
    args.nheads = 8
    args.dim_feedforward = 2048
    args.enc_layers = 6
    args.dec_layers = 6
    args.pre_norm = False
    # query
    args.query_encoding = "b1"
    args.position_embedding = "sine"
    args.dilation = False
    args.num_queries = 10
    args.query_shuffle = False


    return args

if __name__ == "__main__":
    args = parse_args()
    main(args, dataset="mot17_train_FRCNN", load_results=False, frame_range={"start": 0.0, "end": 1.0}, interpolate=False, write_images=False)