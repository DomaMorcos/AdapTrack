import os
import cv2
import pickle
import random
import argparse
import numpy as np
import json
import torch
import torch.nn as nn
import torchreid
from fastreid.emb_computer import EmbeddingComputer  # Your version

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
    """Implementation using EmbeddingComputer."""
    def __init__(self, model_path, config=None):
        super().__init__(model_path, config)
        self.embedder = EmbeddingComputer(dataset=config.get("dataset", "generic"), path=model_path)

    def compute_embedding(self, image, boxes):
        return self.embedder.compute_embedding(image, boxes)

class OSNetExtractor(ReIDExtractor):
    """Implementation using OSNet model."""
    def __init__(self, model_path, config=None):
        super().__init__(model_path, config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OSNetReID(embedding_dim=config.get("embedding_dim", 256))
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.crop_size = (128, 384)

    def compute_embedding(self, image, boxes):
        h, w = image.shape[:2]
        boxes = np.round(boxes).astype(np.int32)
        boxes[:, 0] = boxes[:, 0].clip(0, w)
        boxes[:, 1] = boxes[:, 1].clip(0, h)
        boxes[:, 2] = boxes[:, 2].clip(0, w)
        boxes[:, 3] = boxes[:, 3].clip(0, h)

        crops = []
        for box in boxes:
            crop = image[box[1]:box[3], box[0]:box[2]]
            if crop.size == 0:
                continue
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_LINEAR)
            crop = crop.astype(np.float32) / 255.0
            crop = torch.from_numpy(crop.transpose(2, 0, 1)).unsqueeze(0)
            crops.append(crop)

        if not crops:
            return np.zeros((len(boxes), 256))

        batch = torch.cat(crops, dim=0).to(self.device)
        with torch.no_grad():
            embeddings = self.model(batch)
        
        emb_array = np.zeros((len(boxes), embeddings.shape[1]), dtype=np.float32)
        emb_array[:len(crops)] = embeddings.cpu().numpy()
        return emb_array

class OSNetReID(nn.Module):
    def __init__(self, embedding_dim=256):
        super(OSNetReID, self).__init__()
        self.model = torchreid.models.build_model(
            name='osnet_ain_x1_0',
            num_classes=1000,
            pretrained=True
        )
        self.model.classifier = nn.Identity()
        for name, param in self.model.named_parameters():
            if 'conv4' not in name:
                param.requires_grad = False
        self.fc = nn.Linear(512, embedding_dim)
    
    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)

def make_parser():
    parser = argparse.ArgumentParser("General Feature Extraction")
    parser.add_argument("--pickle_path", type=str, required=True, help="Path to detection pickle file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save detections with features")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with images (e.g., 000001.jpg)")
    parser.add_argument("--reid_model", type=str, default=None, help="Path to ReID model weights")
    parser.add_argument("--reid_class", type=str, default="OSNetExtractor", help="ReID extractor class")
    parser.add_argument("--reid_config", type=str, default='{"dataset": "mot20", "embedding_dim": 256}', help="JSON config for ReID")
    parser.add_argument("--image_ext", type=str, default=".jpg", help="Image file extension")
    parser.add_argument("--frame_padding", type=int, default=6, help="Zero-padding for frame IDs")
    parser.add_argument("--seed", type=int, default=10000, help="Random seed")
    return parser

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    reid_config = json.loads(args.reid_config)
    reid_class = globals()[args.reid_class]
    embedder = reid_class(model_path=args.reid_model, config=reid_config)
    print(f"Using ReID model: {args.reid_model} with class {args.reid_class}")

    with open(args.pickle_path, 'rb') as f:
        detections = pickle.load(f)

    updated_detections = {}
    # Flatten AdapTrack's {vid_name: {frame_id: det_array}} to {frame_id: det_array}
    for vid_name in detections:
        for frame_id, det_array in detections[vid_name].items():
            if det_array is None or det_array.shape[0] == 0:
                updated_detections[int(frame_id)] = det_array
                continue

            img_path = os.path.join(args.image_dir, f"{frame_id:0{args.frame_padding}d}{args.image_ext}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Failed to load {img_path}")
                updated_detections[int(frame_id)] = det_array
                continue

            embedding = embedder.compute_embedding(img, det_array[:, :4])
            if embedding.shape[0] != det_array.shape[0]:
                print(f"Warning: Embedding shape mismatch for frame {frame_id}. Expected {det_array.shape[0]}, got {embedding.shape[0]}")
                continue
            updated_detections[int(frame_id)] = np.concatenate([det_array[:, :5], embedding], axis=1)  # Drop class_id

            print(f"Processed frame {frame_id}", flush=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(updated_detections, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Detections with features saved to {args.output_path}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)