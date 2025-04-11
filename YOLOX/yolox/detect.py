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
from detectors import EnsembleDetector, YoloDetector  # Your BoostTrack++ detectors.py
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

    # Initialize ensemble detector
    detector1 = YoloDetector(model_path=args.model1_path, conf_thresh=args.confthre)
    detector2 = YoloDetector(model_path=args.model2_path, conf_thresh=args.confthre)
    ensemble = EnsembleDetector(
        model1=detector1.model,
        model2=detector2.model,
        model1_weight=args.model1_weight,
        model2_weight=args.model2_weight,
        conf_thresh=args.confthre,
        iou_thresh=args.nmsthre  # Overridden by YOLOX NMS
    )

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

        # Inference
        with torch.no_grad():
            outputs1 = detector1.model(img_tensor)
            outputs2 = detector2.model(img_tensor)
            # Combine predictions (average before NMS)
            outputs = (args.model1_weight * outputs1 + args.model2_weight * outputs2) / (args.model1_weight + args.model2_weight)
            # Apply YOLOX NMS
            outputs = postprocess(outputs, num_classes=1, confthre=args.confthre, nmsthre=args.nmsthre)

        # Process detections
        det = outputs[0]
        if det is not None:
            det[:, 4] *= det[:, 5]  # obj_conf * class_conf
            det[:, 5] = 0  # class_id = 0 (person)
            det = det[:, :6].cpu().numpy()
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