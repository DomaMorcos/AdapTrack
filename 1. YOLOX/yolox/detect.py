import os
import torch
import random
import pickle
import argparse
import numpy as np
import cv2
from detectors import YoloDetector  # Import your custom detector

def make_parser():
    parser = argparse.ArgumentParser("Custom YOLO Detection for MOT")
    parser.add_argument("--dataset_name", type=str, default="tarsh", help="Name of the dataset")
    parser.add_argument("--output_folder", type=str, default="/kaggle/working/results", help="Folder to save results")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to image sequence (e.g., MOT20 test/01/img1)")
    parser.add_argument("--frame_rate", type=int, default=25, help="Frame rate of the sequence")
    parser.add_argument("--yolo_path", type=str, required=True, help="Path to YOLOv12 model weights")
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
            continue
        yield frame_id, img

def main(args):
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    # Initialize detector
    detector = YoloDetector(args.yolo_path)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use first GPU

    # Prepare output directory
    output_dir = os.path.join(args.output_folder, args.dataset_name, "det")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.exp_name)

    # Process sequence and collect detections
    det_results = {}
    for frame_id, img in load_images(args.dataset_path):
        print(f"Processing frame {frame_id}", end="\r")
        with torch.no_grad():
            preds = detector(img)  # [x1, y1, x2, y2, conf]
        
        # Convert to expected format (if needed, adjust based on downstream tracker requirements)
        if preds.shape[0] > 0:
            dets = preds.cpu().numpy()  # [x1, y1, x2, y2, conf]
        else:
            dets = np.zeros((0, 5), dtype=np.float32)
        
        # Store detections (frame_id as key, detections as value)
        det_results[frame_id] = dets

    # Save results as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(det_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nDetections saved to {output_path}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)