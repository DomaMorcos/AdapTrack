import os
import torch
import random
import pickle
import argparse
import numpy as np
import cv2
from detectors import YoloDetector, EnsembleDetector

def make_parser():
    parser = argparse.ArgumentParser("Custom YOLO Ensemble Detection for MOT")
    parser.add_argument("--dataset_name", type=str, default="tarsh", help="Name of the dataset")
    parser.add_argument("--output_folder", type=str, default="/kaggle/working/results", help="Folder to save results")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to image sequence (e.g., MOT20 test/01/img1)")
    parser.add_argument("--frame_rate", type=int, default=25, help="Frame rate of the sequence")
    parser.add_argument("--model1_path", type=str, required=True, help="Path to first YOLOv12 model weights")
    parser.add_argument("--model1_weight", type=float, default=0.5, help="Weight for first model in ensemble")
    parser.add_argument("--model2_path", type=str, required=True, help="Path to second YOLOv12 model weights")
    parser.add_argument("--model2_weight", type=float, default=0.5, help="Weight for second model in ensemble")
    parser.add_argument("--iou_thresh", type=float, default=0.6, help="IoU threshold for WBF")
    parser.add_argument("--conf_thresh", type=float, default=0.3, help="Confidence threshold for detections post-WBF")
    parser.add_argument("--det_thresh", type=float, default=0.4, help="Final detection threshold after boosting (MOT20)")  # Aligned with BoostTrack++
    parser.add_argument("--min_area", type=float, default=100, help="Minimum box area")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="Aspect ratio threshold (width/height)")
    parser.add_argument("--exp_name", type=str, default="detections.pickle", help="Output pickle file name")
    parser.add_argument("--seed", type=int, default=10000, help="Random seed for reproducibility")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--input_size", type=str, default="896,1600", help="Input size for YOLO models (height,width)")  # Aligned with BoostTrack++
    return parser

def preproc(image, input_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), swap=(2, 0, 1)):
    """Preprocess image with ImageNet-like normalization, matching BoostTrack++."""
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image, dtype=np.float32)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]  # BGR to RGB
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def load_images(dataset_path, input_size):
    """Load and preprocess images with ImageNet-like normalization."""
    img_paths = sorted([os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith(('.jpg', '.png'))])
    for frame_id, img_path in enumerate(img_paths, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load {img_path}")
            yield frame_id, None, None, None
            continue
        # Preprocess image
        preprocessed_img, scale = preproc(img, input_size)
        yield frame_id, img, preprocessed_img, scale

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

def soft_biou_batch(boxes1, boxes2):
    """Compute soft buffered IoU (simplified version)."""
    # For simplicity, use regular IoU (BoostTrack++ uses a more complex buffered IoU)
    return iou_batch(boxes1, boxes2)

def shape_similarity(dets, trks):
    """Compute shape similarity between detections and tracks."""
    if dets.shape[0] == 0 or trks.shape[0] == 0:
        return np.zeros((dets.shape[0], trks.shape[0]))
    
    dets_wh = dets[:, 2:4] - dets[:, :2]  # [w, h]
    trks_wh = trks[:, 2:4] - trks[:, :2]  # [w, h]
    
    dets_r = dets_wh[:, 0] / (dets_wh[:, 1] + 1e-6)
    trks_r = trks_wh[:, 0] / (trks_wh[:, 1] + 1e-6)
    
    # Simplified shape similarity (BoostTrack++ uses a corrected version optionally)
    r_diff = np.abs(dets_r[:, None] - trks_r[None, :])
    return np.exp(-r_diff)

def filter_detections(dets, min_area, aspect_ratio_thresh):
    """Filter detections based on area and aspect ratio, matching BoostTrack++ logic."""
    if dets.shape[0] == 0:
        return dets
    widths = dets[:, 2] - dets[:, 0]
    heights = dets[:, 3] - dets[:, 1]
    areas = widths * heights
    aspect_ratios = widths / heights
    vertical = aspect_ratios > aspect_ratio_thresh
    mask = (areas >= min_area) & (~vertical)
    print(f"Filtering: {len(dets)} detections before, {np.sum(mask)} detections after")
    return dets[mask]

def boost_confidence(dets, prev_dets, iou_threshold=0.2, boost_coef=0.5, det_thresh=0.3):
    """Boost confidence of detections that have high IoU with detections in the previous frame."""
    if dets.shape[0] == 0 or prev_dets.shape[0] == 0:
        return dets
    iou_matrix = iou_batch(dets, prev_dets)
    max_iou = iou_matrix.max(axis=1)
    boost_mask = (max_iou > iou_threshold) & (dets[:, 4] < det_thresh)
    dets[boost_mask, 4] = np.maximum(dets[boost_mask, 4], max_iou[boost_mask] * boost_coef)
    return dets

def dlo_confidence_boost(dets, prev_dets, det_thresh=0.4, dlo_boost_coef=0.5, use_rich_sim=True, use_soft_boost=True, use_varying_th=True):
    """Boost detection confidences based on similarity with previous tracks (BoostTrack++ logic)."""
    if dets.shape[0] == 0 or prev_dets.shape[0] == 0:
        return dets
    
    # Simplified Mahalanobis distance (BoostTrack++ uses Kalman filter states, we approximate)
    def mh_dist_matrix(dets, prev_dets):
        centers_dets = (dets[:, :2] + dets[:, 2:4]) / 2
        centers_prev = (prev_dets[:, :2] + prev_dets[:, 2:4]) / 2
        dist = np.sum((centers_dets[:, None] - centers_prev[None, :]) ** 2, axis=2)
        return dist
    
    # Compute similarity matrices
    sbiou_matrix = soft_biou_batch(dets, prev_dets)
    if use_rich_sim:
        mhd_sim = np.exp(-mh_dist_matrix(dets, prev_dets) / 1000)  # Simplified scaling
        shape_sim = shape_similarity(dets, prev_dets)
        S = (mhd_sim + shape_sim + sbiou_matrix) / 3
    else:
        S = iou_batch(dets, prev_dets)

    if not use_soft_boost and not use_varying_th:
        max_s = S.max(1)
        dets[:, 4] = np.maximum(dets[:, 4], max_s * dlo_boost_coef)
    else:
        if use_soft_boost:
            max_s = S.max(1)
            alpha = 0.65
            dets[:, 4] = np.maximum(dets[:, 4], alpha * dets[:, 4] + (1 - alpha) * max_s ** (1.5))
        if use_varying_th:
            threshold_s = 0.95
            threshold_e = 0.8
            n_steps = 20
            alpha = (threshold_s - threshold_e) / n_steps
            # Simplified time_since_update (BoostTrack++ uses tracker states)
            time_since_update = np.zeros(prev_dets.shape[0])  # Placeholder
            tmp = (S > np.maximum(threshold_s - time_since_update[None, :] * alpha, threshold_e)).max(1)
            scores = dets[:, 4].copy()
            scores[tmp] = np.maximum(scores[tmp], det_thresh + 1e-5)
            dets[:, 4] = scores

    return dets

def duo_confidence_boost(dets, prev_dets, det_thresh=0.4):
    """Boost confidences of detections that are close to each other but far from tracks (BoostTrack++ logic)."""
    if dets.shape[0] == 0 or prev_dets.shape[0] == 0:
        return dets
    
    # Simplified Mahalanobis distance (approximation)
    def mh_dist_matrix(dets, prev_dets):
        centers_dets = (dets[:, :2] + dets[:, 2:4]) / 2
        centers_prev = (prev_dets[:, :2] + prev_dets[:, 2:4]) / 2
        dist = np.sum((centers_dets[:, None] - centers_prev[None, :]) ** 2, axis=2)
        return dist
    
    n_dims = 4
    limit = 13.2767
    mahalanobis_distance = mh_dist_matrix(dets, prev_dets)

    if mahalanobis_distance.size > 0:
        min_mh_dists = mahalanobis_distance.min(1)
        mask = (min_mh_dists > limit) & (dets[:, 4] < det_thresh)
        boost_detections = dets[mask]
        boost_detections_args = np.argwhere(mask).reshape((-1,))
        iou_limit = 0.3
        if len(boost_detections) > 0:
            bdiou = iou_batch(boost_detections, boost_detections) - np.eye(len(boost_detections))
            bdiou_max = bdiou.max(axis=1)
            remaining_boxes = boost_detections_args[bdiou_max <= iou_limit]
            args = np.argwhere(bdiou_max > iou_limit).reshape((-1,))
            for i in range(len(args)):
                boxi = args[i]
                tmp = np.argwhere(bdiou[boxi] > iou_limit).reshape((-1,))
                args_tmp = np.append(np.intersect1d(boost_detections_args[args], boost_detections_args[tmp]), boost_detections_args[boxi])
                conf_max = np.max(dets[args_tmp, 4])
                if dets[boost_detections_args[boxi], 4] == conf_max:
                    remaining_boxes = np.array(remaining_boxes.tolist() + [boost_detections_args[boxi]])
            mask = np.zeros_like(dets[:, 4], dtype=np.bool_)
            mask[remaining_boxes] = True
        dets[:, 4] = np.where(mask, det_thresh + 1e-4, dets[:, 4])

    return dets

def visualize_detections(img, dets, output_path, stage):
    """Draw bounding boxes on the image and save it."""
    img_copy = img.copy()
    if dets.shape[0] == 0:
        print(f"No detections to visualize for {stage}")
        return
    
    color = {
        "wbf": (0, 0, 255),      # Red
        "boosting": (0, 255, 0),  # Green
        "filtering": (255, 0, 0)  # Blue
    }[stage]
    
    for det in dets:
        x1, y1, x2, y2, conf = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_copy, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(output_path, img_copy)
    print(f"Saved visualization for {stage} at {output_path}")

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

    # Parse input size
    input_size = tuple(map(int, args.input_size.split(',')))  # (height, width)

    # Initialize detectors and ensemble
    model1 = YoloDetector(args.model1_path, input_size=input_size)  # Pass input size to YoloDetector
    model2 = YoloDetector(args.model2_path, input_size=input_size)
    detector = EnsembleDetector(model1, model2, args.model1_weight, args.model2_weight, args.iou_thresh, args.conf_thresh)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # Prepare output directory for detections and visualizations
    output_dir = os.path.join(args.output_folder, args.dataset_name, "det")
    viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.exp_name)

    # Process sequence and collect detections
    det_results = {}
    prev_dets = np.zeros((0, 5), dtype=np.float32)
    for frame_id, img, preprocessed_img, scale in load_images(args.dataset_path, input_size):
        if img is None:
            det_results[frame_id] = np.zeros((0, 5), dtype=np.float32)
            prev_dets = det_results[frame_id]
            print(f"Frame {frame_id}: No image loaded, 0 detections")
            continue
        
        print(f"Processing frame {frame_id}", end="\r")
        with torch.no_grad():
            # Pass preprocessed image to detector
            preds = detector(preprocessed_img)
            if preds is not None:
                preds = preds.cpu().numpy()
                # Rescale detections back to original image size
                preds[:, :4] /= scale
        
        if preds.shape[0] > 0:
            dets = preds
            print(f"Frame {frame_id}: {len(dets)} detections after WBF")
            
            # Visualize after WBF (every 50 frames)
            if frame_id % 50 == 0:
                viz_path = os.path.join(viz_dir, f"frame_{frame_id}_wbf.jpg")
                visualize_detections(img, dets, viz_path, "wbf")
            
            # Apply BoostTrack++-style confidence boosting
            dets = dlo_confidence_boost(dets, prev_dets, args.det_thresh, dlo_boost_coef=0.5, 
                                      use_rich_sim=True, use_soft_boost=True, use_varying_th=True)
            dets = duo_confidence_boost(dets, prev_dets, args.det_thresh)
            print(f"Frame {frame_id}: {len(dets)} detections after boosting")
            
            # Visualize after boosting (every 50 frames)
            if frame_id % 50 == 0:
                viz_path = os.path.join(viz_dir, f"frame_{frame_id}_boosting.jpg")
                visualize_detections(img, dets, viz_path, "boosting")
            
            # Apply final detection threshold
            dets = dets[dets[:, 4] >= args.det_thresh]
            print(f"Frame {frame_id}: {len(dets)} detections after det_thresh filtering")
            
            dets = filter_detections(dets, args.min_area, args.aspect_ratio_thresh)
            prev_dets = dets
        else:
            dets = np.zeros((0, 5), dtype=np.float32)
            prev_dets = dets
        
        det_results[frame_id] = dets
        print(f"Frame {frame_id}: {len(dets)} detections after filtering")
        
        # Visualize after filtering (every 50 frames)
        if frame_id % 50 == 0:
            viz_path = os.path.join(viz_dir, f"frame_{frame_id}_filtering.jpg")
            visualize_detections(img, dets, viz_path, "filtering")

    # Sort det_results by frame_id to ensure order
    det_results = dict(sorted(det_results.items()))

    # Save results as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(det_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nDetections saved to {output_path}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)