import os
import torch
import random
import pickle
import warnings
import numpy as np
from trackers import metrics
from trackers.tracker import Tracker
from trackers.units import Detection
from interpolation.GSI import gsi_interpolation
from AFLink.AppFreeLink import AFLink
from AFLink.model import PostLinker
from AFLink.dataset import LinkData
import argparse

def make_parser():
    parser = argparse.ArgumentParser("AdapTrack - Tracking with AFLink")
    parser.add_argument("--dataset", type=str, default="MOT20", help="Dataset name (e.g., MOT20)")
    parser.add_argument("--det_feat_path", type=str, required=True, help="Path to detection+feature pickle file")
    parser.add_argument("--save_dir", type=str, default="/kaggle/working/results/tarsh/tracks", help="Directory to save tracking results")
    parser.add_argument("--vid_name", type=str, default="MOT20-01", help="Video sequence name")
    parser.add_argument("--frame_rate", type=int, default=25, help="Frame rate of the sequence (from seqinfo.ini)")
    parser.add_argument("--conf_thresh", type=float, default=0.1, help="Confidence threshold for detections")
    parser.add_argument("--min_box_area", type=float, default=100, help="Minimum box area to keep")
    parser.add_argument("--interpolation", action="store_true", help="Apply GSI interpolation")
    parser.add_argument("--aflink", action="store_true", help="Apply AFLink post-processing")
    parser.add_argument("--aflink_weight_path", type=str, default="/kaggle/working/AdapTrack/AdapTrack/AFLink/AFLink_epoch20.pth", help="Path to AFLink model weights")
    parser.add_argument("--seed", type=int, default=10000, help="Random seed for reproducibility")
    return parser

def create_detections(det_feat):
    detections = []
    if det_feat is None or det_feat.shape[0] == 0:
        return detections
    for row in det_feat:
        bbox, confidence, feature = row[:4], row[4], row[5:]  # [x1, y1, x2, y2, conf, embedding]
        if confidence < opt.conf_thresh:
            continue
        detections.append(Detection(bbox, confidence, feature))
    return detections

def run(vid_name, det_feat, save_path):
    metric = metrics.NearestNeighborDistanceMetric()
    tracker = Tracker(metric, vid_name)
    results = []

    for frame_idx in det_feat.keys():
        detections = create_detections(det_feat[frame_idx])
        tracker.camera_update()  # Placeholder
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            if bbox[2] * bbox[3] > opt.min_box_area and bbox[2] / bbox[3] <= 1.6:
                results.append([frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

        if frame_idx % 50 == 0:
            print(f"{vid_name} {frame_idx} / {len(det_feat.keys())} Finished", flush=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        for row in results:
            print(f"{row[0]},{row[1]},{row[2]:.2f},{row[3]:.2f},{row[4]:.2f},{row[5]:.2f},1,-1,-1,-1", file=f)
    return len(det_feat.keys())

def main():
    args = make_parser().parse_args()
    global opt
    opt = args

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)

    # Initialize AFLink if enabled
    if opt.aflink:
        model = PostLinker()
        model.load_state_dict(torch.load(opt.aflink_weight_path))
        model.eval()
        model.cuda()  # Move to GPU as per AFLink.py
        dataset = LinkData('', '')  # Placeholder; AFLink doesn’t use these paths directly
        print(f"AFLink model loaded from {opt.aflink_weight_path}")

    with open(opt.det_feat_path, 'rb') as f:
        det_feat = pickle.load(f)
        expected_frames = 429  # From seqinfo.ini
        if len(det_feat.keys()) != expected_frames:
            print(f"Warning: Expected {expected_frames} frames, got {len(det_feat.keys())}")

    opt.max_age = opt.frame_rate * 2  # 50 for 25 FPS from seqinfo.ini
    save_path = os.path.join(opt.save_dir, f"{opt.vid_name}.txt")

    start_time = time.time()
    img_num = run(vid_name=opt.vid_name, det_feat=det_feat, save_path=save_path)

    # Post-processing
    sub_time = 0
    if opt.aflink:
        linker = AFLink(path_in=save_path, path_out=save_path, model=model, dataset=dataset,
                        thrT=(0, 30), thrS=75, thrP=0.05)
        sub_time += linker.link()
        print(f"AFLink post-processing completed for {opt.vid_name}")

    if opt.interpolation:
        gsi_interpolation(save_path, save_path, interval=20, tau=10)
        print(f"GSI interpolation completed for {opt.vid_name}")

    total_time = (time.time() - start_time) - sub_time
    time_per_img = total_time / img_num
    print(f"Time per image: {time_per_img:.4f} sec, FPS: {1 / time_per_img:.2f}", flush=True)
# UPDATED
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()