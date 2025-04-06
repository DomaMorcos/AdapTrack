import os
import cv2
import pickle
import random
import argparse
import numpy as np
import json

# Abstract ReID interface
class ReIDExtractor:
    def __init__(self, model_path, config=None):
        self.model_path = model_path
        self.config = config or {}

    def compute_embedding(self, image, boxes):
        """Extract embeddings for given boxes in an image.
        Args:
            image: np.array (H, W, 3) - BGR image
            boxes: np.array (N, 4) - [x1, y1, x2, y2]
        Returns:
            np.array (N, embedding_size) - Feature embeddings
        """
        raise NotImplementedError("Subclasses must implement compute_embedding")

class FastReIDExtractor(ReIDExtractor):
    """Implementation using your EmbeddingComputer."""
    def __init__(self, model_path, config=None):
        super().__init__(model_path, config)
        from emb_computer import EmbeddingComputer  # Local import
        self.embedder = EmbeddingComputer(dataset=config.get("dataset", "generic"), 
                                         path=model_path)

    def compute_embedding(self, image, boxes):
        return self.embedder.compute_embedding(image, boxes)

def make_parser():
    parser = argparse.ArgumentParser("General Feature Extraction")
    parser.add_argument("--pickle_path", type=str, required=True, help="Path to detection pickle file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save detections with features")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with images (e.g., 000001.jpg)")
    parser.add_argument("--reid_model", type=str, default="/kaggle/working/AdapTrack/FastReID/weights/mot17_sbs_S50.pth", help="Path to ReID model weights")
    parser.add_argument("--reid_class", type=str, default="FastReIDExtractor", help="ReID extractor class (e.g., FastReIDExtractor)")
    parser.add_argument("--reid_config", type=str, default='{"dataset": "mot17"}', help="JSON config for ReID (e.g., '{\"dataset\": \"mot17\"}')")
    parser.add_argument("--image_ext", type=str, default=".jpg", help="Image file extension")
    parser.add_argument("--frame_padding", type=int, default=6, help="Zero-padding for frame IDs (e.g., 6 for 000001)")
    parser.add_argument("--seed", type=int, default=10000, help="Random seed")
    return parser

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # Load ReID extractor
    reid_config = json.loads(args.reid_config)
    reid_class = globals()[args.reid_class]  # Assumes class is in this file
    embedder = reid_class(model_path=args.reid_model, config=reid_config)
    print(f"Using ReID model: {args.reid_model} with class {args.reid_class}")

    # Load detections
    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)

    # Feature extraction
    updated_detections = {}
    for frame_id in detections.keys():
        if detections[frame_id].shape[0] == 0:
            updated_detections[frame_id] = detections[frame_id]
            continue

        # Load image with configurable naming
        img_path = os.path.join(args.image_dir, f"{frame_id:0{args.frame_padding}d}{args.image_ext}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load {img_path}")
            updated_detections[frame_id] = detections[frame_id]
            continue

        # Compute embeddings
        embedding = embedder.compute_embedding(img, detections[frame_id][:, :4])
        updated_detections[frame_id] = np.concatenate([detections[frame_id], embedding], axis=1)

        print(f"Processed frame {frame_id}", flush=True)

    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(updated_detections, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Detections with features saved to {args.output_path}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)