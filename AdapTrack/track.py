import os
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
logger.add(sys.stderr, level="DEBUG")

print("Running custom track.py at /kaggle/working/AdapTrack/AdapTrack/track.py")

class InferenceDataset:
    def transform(self, x1, x2):
        if isinstance(x1, list):
            x1 = np.array(x1)
        if isinstance(x2, list):
            x2 = np.array(x2)
        
        min_ = np.concatenate((x1, x2), axis=0).min(axis=0)
        max_ = np.concatenate((x1, x2), axis=0).max(axis=0)
        subtractor = (max_ + min_) / 2
        divisor = (max_ - min_) / 2 + 1e-5
        x1 = (x1 - subtractor) / divisor
        x2 = (x2 - subtractor) / divisor

        x1 = torch.tensor(x1, dtype=torch.float)
        x2 = torch.tensor(x2, dtype=torch.float)
        x1 = x1.unsqueeze(dim=0)
        x2 = x2.unsqueeze(dim=0)
        return x1, x2

def get_color(track_id):
    h = (track_id % 100) / 100.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))

def visualize_tracks(img, tracks, frame_id, stage, output_dir, vis_interval, img_dir, frame_padding='.jpg'):
    if frame_id % vis_interval != 0:
        return
    if not tracks:
        return
    if img is None:
        img_path = os.path.join(img_dir, f"{frame_id:06d}{frame_padding}")
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to load {img_path} for visualization")
            return
    img_vis = img.copy()
    for track in tracks:
        track_id, x, y, w, h, score = track[:6]
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        color = get_color(int(track_id))
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_vis, f"ID:{int(track_id)} s:{score:.2f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{stage}_frame_{frame_id:06d}.jpg"), img_vis)
    logger.info(f"Saved {stage} visualization for frame {frame_id}")

def make_parser():
    parser = argparse.ArgumentParser("AdapTrack Tracking")
    parser.add_argument("--det_feat_path", type=str, required=True, help="Path to detection features pickle")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tracks")
    parser.add_argument("--sequence_name", type=str, required=True, help="Sequence name (e.g., MOT20-01)")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with images (e.g., img1)")
    parser.add_argument("--frame_rate", type=int, default=50, help="Frame rate for max_age")
    parser.add_argument("--post_process", nargs="+", default=["aflink", "interpolation"], help="Post-processing steps")
    parser.add_argument("--conf_thresh", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--ema_beta", type=float, default=0.91, help="EMA beta for feature smoothing")
    parser.add_argument("--min_area", type=float, default=0, help="Minimum box area")
    parser.add_argument("--max_distance", type=float, default=0.45, help="Max distance for tracking")
    parser.add_argument("--max_iou_distance", type=float, default=0.70, help="Max IoU distance")
    parser.add_argument("--min_len", type=int, default=3, help="Minimum track length")
    parser.add_argument("--max_age", type=int, default=None, help="Max age (defaults to frame_rate)")
    parser.add_argument("--vis_interval", type=int, default=10, help="Save visualization every N frames")
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
            results[frame_id] = []
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

        # Visualize initial tracks
        visualize_tracks(None, results[frame_id], frame_id, "initial", 
                         os.path.join(opt.output_dir, "initial_vis"), opt.vis_interval, opt.image_dir)
        logger.debug(f"Frame {frame_id}: Stored {len(results[frame_id])} tracks")
        logger.info(f"Processed frame {frame_id}")

    # Save initial tracks for AFLink
    logger.info("Saving initial tracks for AFLink")
    os.makedirs(opt.output_dir, exist_ok=True)
    initial_output_path = os.path.join(opt.output_dir, f"{opt.sequence_name}_initial.txt")
    with open(initial_output_path, 'w') as f:
        for frame_id in sorted(results.keys(), key=int):
            for track in results[frame_id]:
                f.write(f"{frame_id},{track[0]},{track[1]:.2f},{track[2]:.2f},{track[3]:.2f},{track[4]:.2f},{track[5]:.2f}\n")

    logger.info("Starting post-processing")
    gsi_input_path = initial_output_path
    aflink_results = results.copy()
    if "aflink" in opt.post_process:
        logger.debug("Running AFLink post-processing")
        state_dict = torch.load("/kaggle/working/AdapTrack/AdapTrack/AFLink/AFLink_epoch20.pth", weights_only=True)
        model = PostLinker()
        model.load_state_dict(state_dict)
        model.cuda()
        model.eval()
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
        logger.debug("AFLink post-processing completed")
        gsi_input_path = os.path.join(opt.output_dir, f"{opt.sequence_name}_aflink.txt")
        # Load AFLink results
        aflink_data = np.loadtxt(gsi_input_path, delimiter=',')
        aflink_results = {}
        for row in aflink_data:
            frame_id, track_id, x, y, w, h = row[:6]
            frame_id = int(frame_id)
            if frame_id not in aflink_results:
                aflink_results[frame_id] = []
            aflink_results[frame_id].append([int(track_id), x, y, w, h, 1.0])
        # Visualize AFLink tracks
        for frame_id in aflink_results:
            visualize_tracks(None, aflink_results[frame_id], frame_id, "aflink",
                             os.path.join(opt.output_dir, "aflink_vis"), opt.vis_interval, opt.image_dir)

    if "interpolation" in opt.post_process:
        logger.debug("Running GSI interpolation")
        gsi_output_path = os.path.join(opt.output_dir, f"{opt.sequence_name}_gsi.txt")
        gsi_interpolation(gsi_input_path, gsi_output_path, interval=1000, tau=25)
        gsi_results = np.loadtxt(gsi_output_path, delimiter=',')
        results = {}
        for row in gsi_results:
            frame_id, track_id, x, y, w, h = row[:6]
            frame_id = int(frame_id)
            if frame_id not in results:
                results[frame_id] = []
            results[frame_id].append([int(track_id), x, y, w, h, 1.0])
        logger.debug("GSI interpolation completed")
        # Visualize GSI tracks
        for frame_id in results:
            visualize_tracks(None, results[frame_id], frame_id, "gsi",
                             os.path.join(opt.output_dir, "gsi_vis"), opt.vis_interval, opt.image_dir)

    # Visualize final tracks
    for frame_id in results:
        visualize_tracks(None, results[frame_id], frame_id, "final",
                         os.path.join(opt.output_dir, "final_vis"), opt.vis_interval, opt.image_dir)

    logger.info("Saving final tracks")
    final_output_path = os.path.join(opt.output_dir, f"{opt.sequence_name}.txt")
    with open(final_output_path, 'w') as f:
        for frame_id in sorted(results.keys(), key=int):
            for track in results[frame_id]:
                f.write(f"{frame_id},{track[0]},{track[1]:.2f},{track[2]:.2f},{track[3]:.2f},{track[4]:.2f},{track[5]:.2f},-1,-1,-1\n")
    logger.info(f"Tracks saved to {final_output_path}")

if __name__ == "__main__":
    opt = make_parser().parse_args()
    main(opt)