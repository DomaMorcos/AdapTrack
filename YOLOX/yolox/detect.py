#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import pickle
from loguru import logger
import torch
import cv2
import numpy as np
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess
from detectors import YoloDetector  # Your BoostTrack++ detectors.py
from configparser import ConfigParser

def make_parser():
    parser = argparse.ArgumentParser("YOLOX MOT Detection with Ensemble (No JSON)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to MOT image directory (e.g., /img1)")
    parser.add_argument("--seqinfo_path", type=str, default=None, help="Path to seqinfo.ini (defaults to ../seqinfo.ini)")
    parser.add_argument("--model1_path", type=str, required=True, help="Path to YOLO12l weights")
    parser.add_argument("--model1_weight", type=float, default=0.4, help="Weight for YOLO12l predictions")
    parser.add_argument("--model2_path", type=str, required=True, help="Path to YOLO12x weights")
    parser.add_argument("--model2_weight", type=float, default=0.6, help="Weight for YOLO12x predictions")
    parser.add_argument("--output_folder", type=str, default="./output", help="Output folder for detection pickle")
    parser.add_argument("--exp_name", type=str, default="dets.pickle", help="Output pickle filename")
    parser.add_argument("--confthre", type=float, default=0.4, help="Confidence threshold (MOT20 default)")
    parser.add_argument("--nmsthre", type=float, default=0.8, help="NMS IoU threshold (YOLOX default)")
    parser.add_argument("--img_size", type=str, default="608,1088", help="Input image size (height,width)")
    parser.add_argument("--fp16", action="store_true", help="Use half-precision inference (optional, for speed)")
    return parser

def load_seqinfo(seqinfo_path):
    """Load sequence info from seqinfo.ini."""
    config = ConfigParser()
    config.read(seqinfo_path)
    seq_info = {
        "name": config.get("Sequence", "name"),
        "imDir": config.get("Sequence", "imDir"),
        "frameRate": int(config.get("Sequence", "frameRate")),
        "seqLength": int(config.get("Sequence", "seqLength")),
        "imWidth": int(config.get("Sequence", "imWidth")),
        "imHeight": int(config.get("Sequence", "imHeight")),
        "imExt": config.get("Sequence", "imExt")
    }
    return seq_info

def xyxy2cxcywh(boxes):
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return torch.stack((cx, cy, w, h), dim=1)

def main(args):
    # Parse image size
    img_size = tuple(map(int, args.img_size.split(',')))

    # Determine seqinfo.ini path if not provided
    if args.seqinfo_path is None:
        args.seqinfo_path = os.path.join(os.path.dirname(args.dataset_path), "seqinfo.ini")
    if not os.path.exists(args.seqinfo_path):
        raise FileNotFoundError(f"seqinfo.ini not found at {args.seqinfo_path}")

    # Load sequence info
    seq_info = load_seqinfo(args.seqinfo_path)
    video_name = seq_info["name"]  # e.g., "01"
    seq_length = seq_info["seqLength"]
    orig_size = (seq_info["imHeight"], seq_info["imWidth"])
    im_ext = seq_info["imExt"]

    # Preprocessing (YOLOX logic)
    preproc = ValTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # Initialize detectors
    detector1 = YoloDetector(yolo_path=args.model1_path)
    detector2 = YoloDetector(yolo_path=args.model2_path)

    # Apply FP16 if requested
    if args.fp16:
        logger.info("Using FP16 inference")
        detector1.model.half()
        detector2.model.half()

    # Detection loop
    det_results = {video_name: {}}
    for frame_id in range(1, seq_length + 1):
        # Load image
        img_path = os.path.join(args.dataset_path, f"{frame_id:06d}{im_ext}")
        if not os.path.exists(img_path):
            logger.warning(f"Image {img_path} not found, skipping")
            det_results[video_name][frame_id] = None
            continue

        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to load {img_path}, skipping")
            det_results[video_name][frame_id] = None
            continue

        # Preprocess (YOLOX ValTransform)
        img_tensor, _ = preproc(img, None, img_size)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).cuda()
        if args.fp16:
            img_tensor = img_tensor.half()

        # Convert to BGR numpy array for YoloDetector
        img_np = cv2.resize(img, (img_size[1], img_size[0]))

        # Inference
        with torch.no_grad():
            outputs1 = detector1(img_np)  # [N1, 5]: x1, y1, x2, y2, conf
            outputs2 = detector2(img_np)  # [N2, 5]

            # Convert to YOLOX format: [N, 6] (cx, cy, w, h, obj_conf, class_conf)
            if outputs1.shape[0] > 0:
                outputs1_cxcywh = xyxy2cxcywh(outputs1[:, :4])
                outputs1_yolox = torch.zeros((outputs1.shape[0], 6), dtype=torch.float32)
                outputs1_yolox[:, :4] = outputs1_cxcywh
                outputs1_yolox[:, 4] = outputs1[:, 4]  # obj_conf
                outputs1_yolox[:, 5] = 1.0             # class_conf (person)
            else:
                outputs1_yolox = torch.zeros((0, 6))

            if outputs2.shape[0] > 0:
                outputs2_cxcywh = xyxy2cxcywh(outputs2[:, :4])
                outputs2_yolox = torch.zeros((outputs2.shape[0], 6), dtype=torch.float32)
                outputs2_yolox[:, :4] = outputs2_cxcywh
                outputs2_yolox[:, 4] = outputs2[:, 4]
                outputs2_yolox[:, 5] = 1.0
            else:
                outputs2_yolox = torch.zeros((0, 6))

            # Combine predictions
            combined = torch.cat((outputs1_yolox, outputs2_yolox), dim=0)  # [N1 + N2, 6]
            if combined.shape[0] > 0:
                weights = torch.tensor([args.model1_weight] * outputs1_yolox.shape[0] + 
                                      [args.model2_weight] * outputs2_yolox.shape[0]).cuda()
                outputs = combined * weights.view(-1, 1)  # Weighted sum per detection
                outputs = outputs.unsqueeze(0)  # [1, N, 6]
            else:
                outputs = torch.zeros((1, 0, 6)).cuda()

            # Apply YOLOX NMS
            outputs = postprocess(outputs, num_classes=1, conf_thre=args.confthre, nms_thre=args.nmsthre)
            if outputs[0] is not None:
                outputs = outputs[0]  # [N, 7]
            else:
                outputs = None

        # Process detections
        if outputs is not None:
            det = outputs[:, :6].cpu().numpy()  # [x1, y1, x2, y2, obj_conf, class_conf]
            det[:, 4] *= det[:, 5]  # Combine obj_conf and class_conf
            det = det[:, :5]  # Drop class_conf, keep [x1, y1, x2, y2, conf]
            # Scale back to original resolution
            scale = min(img_size[0] / orig_size[0], img_size[1] / orig_size[1])
            det[:, :4] /= scale
            det_results[video_name][frame_id] = det
        else:
            det_results[video_name][frame_id] = None

        logger.info(f"Processed frame {frame_id} for {video_name}")

    # Save results
    os.makedirs(args.output_folder, exist_ok=True)
    pickle_path = os.path.join(args.output_folder, args.exp_name)
    with open(pickle_path, 'wb') as f:
        pickle.dump(det_results, f)
    logger.info(f"Detections saved to {pickle_path}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)