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
from AFLink.model import PostLinker
from interpolation.GSI import gsi_interpolation
import torch
import cv2
import colorsys

logger.remove()
logger.add(sys.stderr, level="INFO")

print("Running custom track.py at /kaggle/working/AdapTrack/AdapTrack/track.py")

class InferenceDataset:
    def transform(self, x1, x2):
        x1 = np.array(x1) if isinstance(x1, list) else x1
        x2 = np.array(x2) if isinstance(x2, list) else x2
        min_ = np.concatenate((x1, x2), axis=0).min(axis=0)
        max_ = np.concatenate((x1, x2), axis=0).max(axis=0)
        subtractor = (max_ + min_) / 2
        divisor = (max_ - min_) / 2 + 1e-5
        x1 = (x1 - subtractor) / divisor
        x2 = (x2 - subtractor) / divisor
        return torch.tensor(x1, dtype=torch.float).unsqueeze(0), torch.tensor(x2, dtype=torch.float).unsqueeze(0)

def get_color(track_id):
    h = (track_id % 100) / 100.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))

def visualize_xyxy_detections(img, dets, frame_id, output_dir, vis_interval, stage, img_dir, frame_padding='.jpg'):
    if frame_id % vis_interval != 0 or dets is None or len(dets) == 0:
        return
    if img is None:
        img_path = os.path.join(img_dir, f"{frame_id:06d}{frame_padding}")
        img = cv2.imread(img_path)
        if img is None:
            return
    img_vis = img.copy()
    for det in dets:
        x1, y1, x2, y2 = map(int, det[:4])
        score = det[4] if len(det) > 4 else 1.0
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_vis, f"s:{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{stage}_frame_{frame_id:06d}.jpg"), img_vis)

def visualize_tracks(img, tracks, frame_id, stage, output_dir, vis_interval, img_dir, frame_padding='.jpg'):
    if frame_id % vis_interval != 0 or not tracks:
        return
    if img is None:
        img_path = os.path.join(img_dir, f"{frame_id:06d}{frame_padding}")
        img = cv2.imread(img_path)
        if img is None:
            return
    img_vis = img.copy()
    for track in tracks:
        track_id, x1, y1, x2, y2, score = track[:6]
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        color = get_color(int(track_id))
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_vis, f"ID:{int(track_id)} s:{score:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{stage}_frame_{frame_id:06d}.jpg"), img_vis)

def make_parser():
    parser = argparse.ArgumentParser("AdapTrack Tracking")
    parser.add_argument("--det_feat_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sequence_name", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--frame_rate", type=int, default=50)
    parser.add_argument("--post_process", nargs="+", default=["aflink", "interpolation"])
    parser.add_argument("--conf_thresh", type=float, default=0.1)
    parser.add_argument("--ema_beta", type=float, default=0.91)
    parser.add_argument("--min_area", type=float, default=0)
    parser.add_argument("--max_distance", type=float, default=0.45)
    parser.add_argument("--max_iou_distance", type=float, default=0.70)
    parser.add_argument("--min_len", type=int, default=3)
    parser.add_argument("--max_age", type=int, default=None)
    parser.add_argument("--vis_interval", type=int, default=10)
    return parser

def main(opt):
    opt.max_age = opt.frame_rate if opt.max_age is None else opt.max_age
    logger.info(f"Using conf_thresh={opt.conf_thresh}, min_area={opt.min_area}")

    with open(opt.det_feat_path, 'rb') as f:
        det_feat = pickle.load(f)

    if not det_feat or not isinstance(det_feat, dict):
        raise ValueError("Detection pickle file is empty or not a dictionary")
    
    vid_name = opt.sequence_name
    if vid_name not in det_feat:
        raise KeyError(f"Video name '{vid_name}' not found in detection pickle")
    
    frame_data = det_feat[vid_name]
    logger.info(f"Loaded detections for video '{vid_name}' with {len(frame_data)} frames")

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

    # Store original coordinates per track_id
    track_coords = {}  # track_id -> [x1, y1, x2, y2]
    results = {}
    frame_ids = sorted(frame_data.keys(), key=int)

    for frame_id in frame_ids:
        dets = frame_data[frame_id]
        if dets is None or dets.shape[0] == 0:
            tracker.predict()
            tracker.update([])
            results[frame_id] = []
        else:
            if dets.shape[1] <= 5:
                raise ValueError(f"Frame {frame_id}: No features found in detections")

            boxes = dets[:, :4]
            scores = dets[:, 4]
            features = dets[:, 5:]
            logger.info(f"Frame {frame_id}: {len(boxes)} pedestrians before filtering")

            visualize_xyxy_detections(None, dets, frame_id, os.path.join(opt.output_dir, "raw_dets_vis"), 
                                     opt.vis_interval, "raw_dets", opt.image_dir)

            mask = (scores >= opt.conf_thresh) & \
                   ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) >= opt.min_area) & \
                   ((boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1] + 1e-6) <= 5.0)
            boxes = boxes[mask]
            scores = scores[mask]
            features = features[mask]
            logger.info(f"Frame {frame_id}: {len(boxes)} pedestrians after filtering")

            detections = [Detection(bbox, score, feature) for bbox, score, feature in zip(boxes, scores, features)]

            # Store original coordinates for new detections
            det_coords = {i: det.tlbr.tolist() for i, det in enumerate(detections)}

            pre_predict_tracks = []
            for track in tracker.tracks:
                if track.is_confirmed() and track.time_since_update <= 1:
                    bbox = track.to_tlwh()
                    score = track.confidence if hasattr(track, 'confidence') else 1.0
                    coords = track_coords.get(track.track_id, None)
                    if coords:
                        pre_predict_tracks.append([track.track_id] + coords + [score])
                    else:
                        x1, y1, w, h = bbox
                        x2, y2 = x1 + w, y1 + h  # Fixed: Corrected y2 calculation
                        pre_predict_tracks.append([track.track_id, x1, y1, x2, y2, score])
            visualize_tracks(None, pre_predict_tracks, frame_id, "pre_predict", 
                             os.path.join(opt.output_dir, "pre_predict_vis"), opt.vis_interval, opt.image_dir)

            tracker.predict()

            post_predict_tracks = []
            for track in tracker.tracks:
                if track.is_confirmed() and track.time_since_update <= 1:
                    bbox = track.to_tlwh()
                    score = track.confidence if hasattr(track, 'confidence') else 1.0
                    coords = track_coords.get(track.track_id, None)
                    if coords:
                        post_predict_tracks.append([track.track_id] + coords + [score])
                    else:
                        x1, y1, w, h = bbox
                        x2, y2 = x1 + w, y1 + h  # Fixed: Corrected y2 calculation
                        post_predict_tracks.append([track.track_id, x1, y1, x2, y2, score])
            visualize_tracks(None, post_predict_tracks, frame_id, "post_predict", 
                             os.path.join(opt.output_dir, "post_predict_vis"), opt.vis_interval, opt.image_dir)

            matches = tracker.update(detections)

            # Handle case where matches is None
            if matches is not None:
                for det_idx, track_idx in matches:
                    track_id = tracker.tracks[track_idx].track_id
                    track_coords[track_id] = det_coords[det_idx]
            else:
                logger.warning(f"Frame {frame_id}: No matches returned from tracker.update()")

            results[frame_id] = []
            for track in tracker.tracks:
                if track.is_confirmed() and track.time_since_update <= 1:
                    score = track.confidence if hasattr(track, 'confidence') else 1.0
                    coords = track_coords.get(track.track_id, None)
                    if coords:
                        results[frame_id].append([track.track_id] + coords + [score])
                    else:
                        bbox = track.to_tlwh()
                        x1, y1, w, h = bbox
                        x2, y2 = x1 + w, y1 + h  # Fixed: Corrected y2 calculation
                        results[frame_id].append([track.track_id, x1, y1, x2, y2, score])
                        track_coords[track.track_id] = [x1, y1, x2, y2]

            # Prune coordinates for deleted tracks
            active_track_ids = {track.track_id for track in tracker.tracks if not track.is_deleted()}
            track_coords = {tid: coords for tid, coords in track_coords.items() if tid in active_track_ids}

        visualize_tracks(None, results[frame_id], frame_id, "initial", 
                         os.path.join(opt.output_dir, "initial_vis"), opt.vis_interval, opt.image_dir)
        logger.info(f"Processed frame {frame_id}")

    logger.info("Saving initial tracks for AFLink")
    os.makedirs(opt.output_dir, exist_ok=True)
    initial_output_path = os.path.join(opt.output_dir, f"{opt.sequence_name}_initial.txt")
    with open(initial_output_path, 'w') as f:
        for frame_id in sorted(results.keys(), key=int):
            for track in results[frame_id]:
                track_id, x1, y1, x2, y2, score = track
                x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                f.write(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.2f}\n")

    total_tracks = sum(len(tracks) for tracks in results.values())
    if total_tracks == 0:
        logger.info("No tracks generated. Skipping post-processing.")
        final_output_path = os.path.join(opt.output_dir, f"{opt.sequence_name}.txt")
        with open(final_output_path, 'w') as f:
            pass
        return

    logger.info("Starting post-processing")
    gsi_input_path = initial_output_path
    aflink_results = {}
    if "aflink" in opt.post_process:
        state_dict = torch.load("/kaggle/working/AdapTrack/AdapTrack/AFLink/AFLink_epoch20.pth", weights_only=True)
        model = PostLinker().cuda().eval()
        model.load_state_dict(state_dict)
        dataset = InferenceDataset()
        aflink = AFLink(
            path_in=initial_output_path,
            path_out=os.path.join(opt.output_dir, f"{opt.sequence_name}_aflink.txt"),
            model=model,
            dataset=dataset,
            thrT=(1, 30),
            thrS=0.4,
            thrP=0.5
        )
        aflink.link()
        gsi_input_path = os.path.join(opt.output_dir, f"{opt.sequence_name}_aflink.txt")
        aflink_data = np.loadtxt(gsi_input_path, delimiter=',')
        for row in aflink_data:
            frame_id, track_id, x, y, w, h = row[:6]
            frame_id = int(frame_id)
            x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
            if frame_id not in aflink_results:
                aflink_results[frame_id] = []
            aflink_results[frame_id].append([int(track_id), x1, y1, x2, y2, 1.0])
        for frame_id in aflink_results:
            visualize_tracks(None, aflink_results[frame_id], frame_id, "aflink",
                             os.path.join(opt.output_dir, "aflink_vis"), opt.vis_interval, opt.image_dir)

    if "interpolation" in opt.post_process:
        gsi_output_path = os.path.join(opt.output_dir, f"{opt.sequence_name}_gsi.txt")
        gsi_interpolation(gsi_input_path, gsi_output_path, interval=1000, tau=25)
        gsi_results = np.loadtxt(gsi_output_path, delimiter=',')
        results = {}
        for row in gsi_results:
            frame_id, track_id, x, y, w, h = row[:6]
            frame_id = int(frame_id)
            x1, y1, x2, y2 = x - w/2, y - h/2, x + w/2, y + h/2
            if frame_id not in results:
                results[frame_id] = []
            results[frame_id].append([int(track_id), x1, y1, x2, y2, 1.0])
        for frame_id in results:
            visualize_tracks(None, results[frame_id], frame_id, "gsi",
                             os.path.join(opt.output_dir, "gsi_vis"), opt.vis_interval, opt.image_dir)

    for frame_id in results:
        visualize_tracks(None, results[frame_id], frame_id, "final",
                         os.path.join(opt.output_dir, "final_vis"), opt.vis_interval, opt.image_dir)

    logger.info("Saving final tracks")
    final_output_path = os.path.join(opt.output_dir, f"{opt.sequence_name}.txt")
    with open(final_output_path, 'w') as f:
        for frame_id in sorted(results.keys(), key=int):
            for track in results[frame_id]:
                track_id, x1, y1, x2, y2, score = track
                x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                f.write(f"{frame_id},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.2f},-1,-1,-1\n")

if __name__ == "__main__":
    opt = make_parser().parse_args()
    main(opt)