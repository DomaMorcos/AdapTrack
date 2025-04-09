import os
import torch
import random
import pickle
import warnings
import numpy as np
import argparse
import time
from trackers import metrics
from trackers.tracker import Tracker
from trackers.units import Detection
from GBI import GBInterpolation
from AFLink.AppFreeLink import AFLink
from AFLink.model import PostLinker
from AFLink.dataset import LinkData

def make_parser():
    parser = argparse.ArgumentParser("General Tracking Pipeline")
    parser.add_argument("--det_feat_path", type=str, required=True, help="Path to detection+feature pickle file")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/results/tracks", help="Directory to save tracking results")
    parser.add_argument("--sequence_name", type=str, default="sequence", help="Name of the sequence")
    parser.add_argument("--frame_rate", type=float, default=25.0, help="Frame rate for max_age")
    parser.add_argument("--min_area", type=float, default=10, help="Minimum box area")  # Aligned with BoostTrack++ (was 100)
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="Aspect ratio threshold (width/height)")  # Already aligned
    parser.add_argument("--max_distance", type=float, default=0.2, help="Max distance for appearance matching")
    parser.add_argument("--max_iou_distance", type=float, default=0.85, help="Max IoU distance for matching")
    parser.add_argument("--min_len", type=int, default=2, help="Minimum hits to confirm track")
    parser.add_argument("--ema_beta", type=float, default=0.92, help="EMA smoothing factor")
    parser.add_argument("--post_process", nargs="*", default=["aflink", "interpolation"], help="Post-processors (e.g., aflink interpolation)")
    parser.add_argument("--aflink_weights", type=str, default="/kaggle/working/AdapTrack/AdapTrack/AFLink/AFLink_epoch20.pth", help="Path to AFLink weights")
    parser.add_argument("--aflink_thrT_min", type=int, default=5, help="Min time gap for AFLink linking")
    parser.add_argument("--aflink_thrT_max", type=int, default=30, help="Max time gap for AFLink linking")
    parser.add_argument("--aflink_thrS", type=int, default=50, help="Spatial threshold for AFLink")
    parser.add_argument("--aflink_thrP", type=float, default=0.01, help="Probability threshold for AFLink")
    parser.add_argument("--seed", type=int, default=10000, help="Random seed")
    return parser

def create_detections(det_feat):
    """Create detections without additional confidence filtering (already applied in generate_dets_feat.py)."""
    detections = []
    if det_feat is None or det_feat.shape[0] == 0:
        print("Debug: No detections in frame")
        return detections
    for row in det_feat:
        bbox, confidence, feature = row[:4], row[4], row[5:]
        detections.append(Detection(bbox, confidence, feature))
    return detections

def run_tracker(sequence_name, det_feat, output_path, args):
    metric = metrics.NearestNeighborDistanceMetric()
    tracker = Tracker(metric, sequence_name, max_age=int(args.frame_rate * 2), 
                      max_distance=args.max_distance, max_iou_distance=args.max_iou_distance, 
                      min_len=args.min_len, ema_beta=args.ema_beta)
    results = []

    # Process all frames, even those with no detections
    total_frames = max(det_feat.keys())
    for frame_idx in range(1, total_frames + 1):
        # Ensure frame exists in det_feat, even if empty
        if frame_idx not in det_feat or len(det_feat[frame_idx]) == 0:
            print(f"Debug: No detections in frame {frame_idx}")
            detections = []
        else:
            detections = create_detections(det_feat[frame_idx])

        tracker.camera_update(frame_idx)
        tracker.predict()
        tracker.update(detections)

        # Collect tracks for this frame
        frame_results = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            # Apply BoostTrack++-style filtering
            vertical = (bbox[2] / bbox[3]) > args.aspect_ratio_thresh
            if bbox[2] * bbox[3] > args.min_area and not vertical:
                frame_results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3], 1.0])

        results.extend(frame_results)

        if frame_idx % 50 == 0:
            print(f"{sequence_name} {frame_idx} / {total_frames} Finished", flush=True)

    # Write initial results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for row in results:
            print(f"{row[0]},{row[1]},{row[2]:.2f},{row[3]:.2f},{row[4]:.2f},{row[5]:.2f},{row[6]:.2f}", file=f)
    return total_frames

def apply_post_processing(output_path, post_processors, args):
    sub_time = 0
    if "aflink" in post_processors:
        model = PostLinker()
        model.load_state_dict(torch.load(args.aflink_weights))
        model.eval()
        model.cuda()
        dataset = LinkData('', '')
        linker = AFLink(path_in=output_path, path_out=output_path, model=model, dataset=dataset,
                        thrT=(args.aflink_thrT_min, args.aflink_thrT_max), thrS=args.aflink_thrS, thrP=args.aflink_thrP)
        sub_time += linker.link()
        print(f"AFLink post-processing completed")

    if "interpolation" in post_processors:
        GBInterpolation(path_in=output_path, path_out=output_path, interval=20)
        sub_time += 0.1  # Placeholder timing
        print(f"GBInterpolation post-processing completed")

    # Sort the output file by frame number
    with open(output_path, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(',') for line in lines if line.strip()]
    lines.sort(key=lambda x: int(x[0]))  # Sort by frame number
    with open(output_path, 'w') as f:
        for line in lines:
            f.write(','.join(line) + '\n')

    return sub_time

def main():
    args = make_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    with open(args.det_feat_path, 'rb') as f:
        det_feat = pickle.load(f)
        print(f"Loaded {len(det_feat)} frames from {args.det_feat_path}")

    output_path = os.path.join(args.output_dir, f"{args.sequence_name}.txt")

    start_time = time.time()
    img_num = run_tracker(args.sequence_name, det_feat, output_path, args)

    sub_time = apply_post_processing(output_path, args.post_process, args)

    total_time = (time.time() - start_time) - sub_time
    time_per_img = total_time / img_num
    print(f"Time per image: {time_per_img:.4f} sec, FPS: {1 / time_per_img:.2f}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()