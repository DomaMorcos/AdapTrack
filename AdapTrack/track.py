import os
import sys
import pickle
import argparse
import numpy as np
from loguru import logger
from trackers.tracker import Tracker
from trackers.metrics import NearestNeighborDistanceMetric
from trackers.units import Detection
from AFLink.AppFreeLink import AFLink
from interpolation.GSI import gsi_interpolation as GSI

logger.remove()
logger.add(sys.stderr, level="DEBUG")

print("Running custom track.py at /kaggle/working/AdapTrack/AdapTrack/track.py")

def make_parser():
    parser = argparse.ArgumentParser("AdapTrack Tracking")
    parser.add_argument("--det_feat_path", type=str, required=True, help="Path to detection features pickle")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tracks")
    parser.add_argument("--sequence_name", type=str, required=True, help="Sequence name (e.g., MOT20-01)")
    parser.add_argument("--frame_rate", type=int, default=50, help="Frame rate for max_age")
    parser.add_argument("--post_process", nargs="+", default=["aflink", "interpolation"], help="Post-processing steps")
    parser.add_argument("--conf_thresh", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--ema_beta", type=float, default=0.91, help="EMA beta for feature smoothing")
    parser.add_argument("--min_area", type=float, default=0, help="Minimum box area")
    parser.add_argument("--max_distance", type=float, default=0.45, help="Max distance for tracking")
    parser.add_argument("--max_iou_distance", type=float, default=0.70, help="Max IoU distance")
    parser.add_argument("--min_len", type=int, default=3, help="Minimum track length")
    parser.add_argument("--max_age", type=int, default=None, help="Max age (defaults to frame_rate)")
    return parser

def main(opt):
    if opt.max_age is None:
        opt.max_age = opt.frame_rate

    logger.info(f"Using conf_thresh={opt.conf_thresh}, min_area={opt.min_area}")

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
        conf_thresh=opt.conf_thresh
    )

    results = {}
    frame_ids = sorted(det_feat.keys(), key=int)
    for frame_id in frame_ids:
        dets = det_feat[frame_id]
        logger.debug(f"Processing frame {frame_id}: {dets.shape if dets is not None else 'None'} detections")
        if dets is None or dets.shape[0] == 0:
            logger.debug(f"Frame {frame_id}: Predicting with no detections")
            tracker.predict()
            logger.debug(f"Frame {frame_id}: Updating with empty detections")
            tracker.update([])
            logger.info(f"Processed frame {frame_id}")
        else:
            if dets.shape[1] <= 5:
                raise ValueError(f"Frame {frame_id}: No features found in detections (shape {dets.shape})")

            boxes = dets[:, :4]
            scores = dets[:, 4]
            features = dets[:, 5:]
            logger.debug(f"Frame {frame_id}: {len(boxes)} detections before filtering")

            mask = (scores >= opt.conf_thresh) & \
                   ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) >= opt.min_area) & \
                   ((boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1] + 1e-6) <= 5.0)
            boxes = boxes[mask]
            scores = scores[mask]
            features = features[mask]
            logger.debug(f"Frame {frame_id}: {len(boxes)} detections after filtering")

            detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(boxes, scores, features)]
            logger.debug(f"Frame {frame_id}: Created {len(detections)} Detection objects")

            logger.debug(f"Frame {frame_id}: Predicting")
            tracker.predict()
            logger.debug(f"Frame {frame_id}: Updating with {len(detections)} detections")
            tracker.update(detections)

        results[frame_id] = []
        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                bbox = track.to_tlwh()
                score = track.confidence if hasattr(track, 'confidence') else 1.0
                results[frame_id].append([track.track_id] + bbox.tolist() + [score])
        logger.debug(f"Frame {frame_id}: Stored {len(results[frame_id])} tracks")
        logger.info(f"Processed frame {frame_id}")

    logger.info("Starting post-processing")
    if "aflink" in opt.post_process:
        logger.debug("Running AFLink post-processing")
        aflink = AFLink(opt.sequence_name, results, model="/kaggle/working/AdapTrack/AdapTrack/AFLink/AFLink_epoch20.pth", dataset="MOT20", thrT=30, thrS=0.4, thrP=0.5)
        results = aflink.process()
        logger.debug("AFLink post-processing completed")

    if "interpolation" in opt.post_process:
        logger.debug("Running GSI interpolation")
        gsi = GSI(opt.sequence_name, results, interval=1000, tau=25)
        results = gsi.process()
        logger.debug("GSI interpolation completed")

    logger.info("Saving tracks")
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