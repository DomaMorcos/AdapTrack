import os
import cv2
import pickle
import random
import argparse
import numpy as np
from fastreid.emb_computer import EmbeddingComputer

def make_parser():
    parser = argparse.ArgumentParser("Track - Feature Extraction")
    # Data args
    parser.add_argument("--dataset", type=str, default="mot20", help="Dataset name (e.g., mot20)")
    parser.add_argument("--pickle_path", type=str, required=True, help="Path to detection pickle file from detect.py")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save detections with features")
    parser.add_argument("--data_path", type=str, required=True, help="Path to image sequence (e.g., /kaggle/input/mot20fawry/tracking/test/01/img1)")
    # Model args
    parser.add_argument("--weights_path", type=str, default="/kaggle/working/AdapTrack/FastReID/weights/mot17_sbs_S50.pth", help="Path to FastReID weights")
    # Else
    parser.add_argument("--seed", type=int, default=10000, help="Random seed for reproducibility")
    return parser

def main(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Print paths for debugging
    print(f"Using weights: {args.weights_path}")

    # Initialize embedding computer with dataset and custom weights path
    embedder = EmbeddingComputer(dataset=args.dataset, path=args.weights_path)

    # Read detection pickle file
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)

    # Feature extraction
    updated_detections = {}
    for frame_id in detections.keys():
        # Skip if no detections
        if detections[frame_id].shape[0] == 0:
            updated_detections[frame_id] = detections[frame_id]
            continue

        # Read image (adjusted for Kaggle path, assuming 6-digit frame IDs)
        img_path = os.path.join(args.data_path, f"{frame_id:06d}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load {img_path}")
            updated_detections[frame_id] = detections[frame_id]  # Keep as-is if image fails
            continue

        # Get detections for this frame
        detection = detections[frame_id]  # [N, 5] array: [x1, y1, x2, y2, conf]

        # Compute embeddings
        embedding = embedder.compute_embedding(img, detection[:, :4])  # Pass [x1, y1, x2, y2]
        updated_dets = np.concatenate([detection, embedding], axis=1)  # [x1, y1, x2, y2, conf, embedding]

        # Store updated detections
        updated_detections[frame_id] = updated_dets

        # Logging
        print(f"Processed frame {frame_id}", flush=True)

    # Save updated detections
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(updated_detections, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Detections with features saved to {args.output_path}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)