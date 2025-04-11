import os
import pickle
import argparse
import numpy as np
from loguru import logger
import torch
import random
import pickle
import warnings
from opts import *
from os.path import join
from trackers import metrics
from AFLink.AppFreeLink import *
from trackeval.run import evaluate
from AFLink.model import PostLinker
from AFLink.dataset import LinkData
from trackers.tracker import Tracker
from trackers.units import Detection
from interpolation.GSI import gsi_interpolation

def make_parser():
    parser = argparse.ArgumentParser("AdapTrack Tracking")
    parser.add_argument("--det_feat_path", type=str, required=True, help="Path to detection features pickle")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tracks")
    parser.add_argument("--sequence_name", type=str, required=True, help="Sequence name (e.g., MOT20-01)")
    parser.add_argument("--frame_rate", type=int, default=50, help="Frame rate for max_age")
    parser.add_argument("--post_process", nargs="+", default=["aflink", "interpolation"], help="Post-processing steps")
    parser.add_argument("--conf_thresh", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--ema_beta", type=float, default=0.91, help="EMA beta for feature smoothing")
    parser.add_argument("--min_area", type=float, default=10, help="Minimum box area")
    parser.add_argument("--max_distance", type=float, default=0.45, help="Max distance for tracking")
    parser.add_argument("--max_iou_distance", type=float, default=0.70, help="Max IoU distance")
    parser.add_argument("--min_len", type=int, default=3, help="Minimum track length")
    parser.add_argument("--max_age", type=int, default=None, help="Max age (defaults to frame_rate)")
    parser.add_argument("--gating_lambda", type=float, default=0.98, help="Gating lambda")
    return parser

def main(opt):
    if opt.max_age is None:
        opt.max_age = opt.frame_rate

    with open(opt.det_feat_path, 'rb') as f:
        det_feat = pickle.load(f)

    logger.info(f"Loaded detections with {len(det_feat)} frames")
    sample_frame = next((fid for fid, dets in det_feat.items() if dets is not None), None)
    if sample_frame:
        logger.info(f"Sample frame {sample_frame}: {det_feat[sample_frame].shape} detections")

    metric = NearestNeighborDistanceMetric()
    tracker = Tracker(
        metric=metric,
        vid_name=opt.sequence_name,
        max_distance=opt.max_distance,
        max_iou_distance=opt.max_iou_distance,
        min_len=opt.min_len,
        max_age=opt.max_age,
        ema_beta=opt.ema_beta,
        gating_lambda=opt.gating_lambda
    )

    results = {}
    frame_ids = sorted(det_feat.keys(), key=int)
    for frame_id in frame_ids:
        dets = det_feat[frame_id]
        if dets is None or dets.shape[0] == 0:
            tracker.predict()
            tracker.update(np.empty((0, 5)), np.empty((0, 0)))
            logger.info(f"Frame {frame_id}: No detections")
        else:
            if dets.shape[1] <= 5:
                raise ValueError(f"Frame {frame_id}: No features found in detections (shape {dets.shape})")

            boxes = dets[:, :4]
            scores = dets[:, 4]
            features = dets[:, 5:]
            mask = (scores >= opt.conf_thresh) & \
                   ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) >= opt.min_area) & \
                   ((boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1] + 1e-6) <= 1.6)
            boxes = boxes[mask]
            scores = scores[mask]
            features = features[mask]

            tracker.predict()
            tracker.update(boxes, features)

        results[frame_id] = []
        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                bbox = track.to_tlwh()
                score = scores[track.last_idx] if track.last_idx >= 0 and track.last_idx < len(scores) else 1.0
                results[frame_id].append([track.track_id] + bbox.tolist() + [score])

        logger.info(f"Processed frame {frame_id}")

    if "aflink" in opt.post_process:
        aflink = AFLink(opt.sequence_name, results, interval=opt.max_age)
        results = aflink.process()

    if "interpolation" in opt.post_process:
        gsi = GSI(opt.sequence_name, results, interval=1000, tau=25)
        results = gsi.process()

    os.makedirs(opt.output_dir, exist_ok=True)
    output_path = os.path.join(opt.output_dir, f"{opt.sequence_name}.txt")
    with open(output_path, 'w') as f:
        for frame_id in sorted(results.keys(), key=int):
            for track in results[frame_id]:
                f.write(f"{frame_id},{track[0]},{track[1]:.2f},{track[2]:.2f},{track[3]:.2f},{track[4]:.2f},{track[5]:.2f},-1,-1,-1\n")
    logger.info(f"Tracks saved to {output_path}")

if __name__ == "__main__":
    opt = make_parser().parse_args()
    main(opt)