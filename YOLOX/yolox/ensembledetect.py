import os
import torch
import random
import pickle
import argparse
import numpy as np
import cv2
from detectors import YoloDetector, EnsembleDetector  # Assumes detectors.py is in the same directory

def make_parser():
    parser = argparse.ArgumentParser("Custom YOLO Ensemble Detection for MOT")
    parser.add_argument("--dataset_name", type=str, default="tarsh", help="Name of the dataset")
    parser.add_argument("--output_folder", type=str, default="/kaggle/working/results", help="Folder to save results")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to image sequence (e.g., MOT20 test/01/img1)")
    parser.add_argument("--frame_rate", type=int, default=25, help="Frame rate of the sequence")
    parser.add_argument("--model1_path", type=str, required=True, help="Path to first YOLOv12 model weights")
    parser.add_argument("--model1_weight", type=float, default=0.7, help="Weight for first model in ensemble")  # Increased
    parser.add_argument("--model2_path", type=str, required=True, help="Path to second YOLOv12 model weights")
    parser.add_argument("--model2_weight", type=float, default=0.3, help="Weight for second model in ensemble")  # Decreased
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="IoU threshold for WBF")  # Lowered
    parser.add_argument("--conf_thresh", type=float, default=0.05, help="Confidence threshold for detections post-WBF")  # Lowered
    parser.add_argument("--min_area", type=float, default=25, help="Minimum box area")  # Lowered
    parser.add_argument("--max_aspect_ratio", type=float, default=3.0, help="Maximum width/height ratio")  # Increased
    parser.add_argument("--exp_name", type=str, default="detections.pickle", help="Output pickle file name")
    parser.add_argument("--seed", type=int, default=10000, help="Random seed for reproducibility")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    return parser

def load_images(dataset_path):
    """Load images from the dataset path in sorted order."""
    img_paths = sorted([os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith(('.jpg', '.png'))])
    for frame_id, img_path in enumerate(img_paths, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load {img_path}")
            yield frame_id, None
            continue
        yield frame_id, img

def iou_batch(boxes1, boxes2):
    """Compute IoU between two sets of boxes."""
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]))
    
    boxes1 = boxes1[:, :4]
    boxes2 = boxes2[:, :4]
    
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - intersection
    
    return intersection / (union + 1e-6)

def filter_detections(dets, min_area, max_aspect_ratio):
    """Filter detections based on area and aspect ratio."""
    if dets.shape[0] == 0:
        return dets
    widths = dets[:, 2] - dets[:, 0]
    heights = dets[:, 3] - dets[:, 1]
    areas = widths * heights
    aspect_ratios = widths / heights
    mask = (areas >= min_area) & (aspect_ratios <= max_aspect_ratio) & (1 / aspect_ratios <= max_aspect_ratio)
    return dets[mask]

def boost_confidence(dets, prev_dets, iou_threshold=0.3, boost_coef=0.5, det_thresh=0.05):  # Aligned with conf_thresh
    """Boost confidence of detections that have high IoU with detections in the previous frame."""
    if dets.shape[0] == 0 or prev_dets.shape[0] == 0:
        return dets
    iou_matrix = iou_batch(dets, prev_dets)
    max_iou = iou_matrix.max(axis=1)
    boost_mask = (max_iou > iou_threshold) & (dets[:, 4] < det_thresh)
    dets[boost_mask, 4] = np.maximum(dets[boost_mask, 4], max_iou[boost_mask] * boost_coef)
    return dets

def main(args):
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    torch.backends.cudnn.benchmark = True

    # Initialize detectors and ensemble
    model1 = YoloDetector(args.model1_path)
    model2 = YoloDetector(args.model2_path)
    detector = EnsembleDetector(model1, model2, args.model1_weight, args.model2_weight, args.iou_thresh, args.conf_thresh)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # Prepare output directory
    output_dir = os.path.join(args.output_folder, args.dataset_name, "det")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.exp_name)

    # Process sequence and collect detections
    det_results = {}
    prev_dets = np.zeros((0, 5), dtype=np.float32)
    for frame_id, img in load_images(args.dataset_path):
        if img is None:
            det_results[frame_id] = np.zeros((0, 5), dtype=np.float32)
            prev_dets = det_results[frame_id]
            print(f"Frame {frame_id}: No image loaded, 0 detections")
            continue
        
        print(f"Processing frame {frame_id}", end="\r")
        with torch.no_grad():
            preds = detector(img)
        
        if preds.shape[0] > 0:
            dets = preds.cpu().numpy()
            print(f"Frame {frame_id}: {len(dets)} detections after WBF")
            dets = boost_confidence(dets, prev_dets, iou_threshold=0.3, boost_coef=0.5, det_thresh=args.conf_thresh)
            print(f"Frame {frame_id}: {len(dets)} detections after boosting")
            dets = filter_detections(dets, args.min_area, args.max_aspect_ratio)
            prev_dets = dets
        else:
            dets = np.zeros((0, 5), dtype=np.float32)
            prev_dets = dets
        
        det_results[frame_id] = dets
        print(f"Frame {frame_id}: {len(dets)} detections after filtering")

    # Sort det_results by frame_id to ensure order
    det_results = dict(sorted(det_results.items()))

    # Save results as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(det_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nDetections saved to {output_path}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)