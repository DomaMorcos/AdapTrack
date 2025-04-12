#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import pickle
from loguru import logger
import torch
import cv2
import numpy as np
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess
from detectors import YoloDetector, EnsembleDetector
from configparser import ConfigParser
import threading

def make_parser():
    parser = argparse.ArgumentParser("YOLOX MOT Detection with Ensemble")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to MOT image directory (e.g., /img1)")
    parser.add_argument("--seqinfo_path", type=str, default=None, help="Path to seqinfo.ini (defaults to ../seqinfo.ini)")
    parser.add_argument("--model1_path", type=str, required=True, help="Path to YOLO12l weights")
    parser.add_argument("--model1_weight", type=float, default=0.4, help="Weight for YOLO12l predictions")
    parser.add_argument("--model2_path", type=str, required=True, help="Path to YOLO12x weights")
    parser.add_argument("--model2_weight", type=float, default=0.6, help="Weight for YOLO12x predictions")
    parser.add_argument("--output_folder", type=str, default="./output", help="Output folder for detection pickle")
    parser.add_argument("--exp_name", type=str, default="dets.pickle", help="Output pickle filename")
    parser.add_argument("--confthre", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--nmsthre", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--img_size", type=str, default="608,1088", help="Input image size (height,width)")
    parser.add_argument("--fp16", action="store_true", help="Use half-precision inference")
    parser.add_argument("--vis_interval", type=int, default=10, help="Save visualization every N frames")
    return parser

def load_seqinfo(seqinfo_path):
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
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = (boxes[:, 2] - boxes[:, 0])
    h = (boxes[:, 3] - boxes[:, 1])
    return torch.stack((cx, cy, w, h), dim=1)

def visualize_xyxy_detections(img, dets, frame_id, output_dir, vis_interval, stage_name, img_size=None):
    if frame_id % vis_interval != 0:
        return
    if dets is None or len(dets) == 0:
        logger.info(f"No detections to visualize at stage {stage_name} for frame {frame_id}")
        return
    # Resize image to match the detection space if specified
    if img_size is not None:
        vis_img = cv2.resize(img, (img_size[1], img_size[0]))
    else:
        vis_img = img.copy()
    for i, det in enumerate(dets):
        x1, y1, x2, y2 = det[:4].cpu().numpy().astype(int)
        score = det[4].cpu().numpy() if len(det) > 4 else 0.0
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        logger.debug(f"{stage_name} detection {i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}, score={score:.2f}")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_id:06d}_{stage_name}.jpg"), vis_img)
    logger.info(f"Saved {stage_name} visualization for frame {frame_id}")

def visualize_cxcywh_detections(img, dets, frame_id, output_dir, vis_interval, stage_name, img_size=None):
    if frame_id % vis_interval != 0:
        return
    if dets is None or len(dets) == 0:
        logger.info(f"No detections to visualize at stage {stage_name} for frame {frame_id}")
        return
    # Resize image to match the detection space if specified
    if img_size is not None:
        vis_img = cv2.resize(img, (img_size[1], img_size[0]))
    else:
        vis_img = img.copy()
    for i, det in enumerate(dets):
        cx, cy, w, h = det[:4].cpu().numpy().astype(int)
        score = det[4].cpu().numpy() if len(det) > 4 else 0.0
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        logger.debug(f"{stage_name} detection {i}: cx={cx}, cy={cy}, w={w}, h={h}, score={score:.2f}")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_id:06d}_{stage_name}.jpg"), vis_img)
    logger.info(f"Saved {stage_name} visualization for frame {frame_id}")

def run_detector(detector, img_tensor_np, output_list, index):
    with torch.no_grad():
        output_list[index] = detector(img_tensor_np)

def main(args):
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, running on CPU")

    img_size = tuple(map(int, args.img_size.split(',')))

    if args.seqinfo_path is None:
        args.seqinfo_path = os.path.join(os.path.dirname(args.dataset_path), "seqinfo.ini")
    if not os.path.exists(args.seqinfo_path):
        raise FileNotFoundError(f"seqinfo.ini not found at {args.seqinfo_path}")

    seq_info = load_seqinfo(args.seqinfo_path)
    video_name = seq_info["name"]
    seq_length = seq_info["seqLength"]
    orig_size = (seq_info["imHeight"], seq_info["imWidth"])
    im_ext = seq_info["imExt"]
    logger.info(f"Original image size: {orig_size}, Resized image size: {img_size}")

    preproc = ValTransform(rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    detector1 = YoloDetector(yolo_path=args.model1_path)
    detector2 = YoloDetector(yolo_path=args.model2_path)
    ensemble_detector = EnsembleDetector(
        model1=detector1,
        model2=detector2,
        model1_weight=args.model1_weight,
        model2_weight=args.model2_weight,
        iou_thresh=0.6,
        conf_thresh=0.3
    )

    if args.fp16:
        logger.info("Using FP16 inference")
        detector1.model = detector1.model.cuda().half()
        detector2.model = detector2.model.cuda().half()
        torch.cuda.synchronize()
    else:
        detector1.model = detector1.model.cuda()
        detector2.model = detector2.model.cuda()

    logger.info(f"Detector1 device: {next(detector1.model.parameters()).device}")
    logger.info(f"Detector2 device: {next(detector2.model.parameters()).device}")

    det_results = {video_name: {}}
    vis_dir = os.path.join(args.output_folder, "det_vis")
    for frame_id in range(1, seq_length + 1):
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

        img_tensor, _ = preproc(img, None, img_size)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).cuda()
        if args.fp16:
            img_tensor = img_tensor.half()

        img_np = cv2.resize(img, (img_size[1], img_size[0]))
        img_tensor_np = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).cuda() / 255.0
        if args.fp16:
            img_tensor_np = img_tensor_np.half()

        # Debug input tensor range
        print(f"Frame {frame_id} - img_tensor_np min: {img_tensor_np.min().item()}, max: {img_tensor_np.max().item()}")

        with torch.no_grad():
            if frame_id == 1:  # Warm-up
                _ = detector1(img_tensor_np)
                _ = detector2(img_tensor_np)
                torch.cuda.synchronize()

            # Run detectors in parallel
            outputs_list = [None, None]
            thread1 = threading.Thread(target=run_detector, args=(detector1, img_tensor_np, outputs_list, 0))
            thread2 = threading.Thread(target=run_detector, args=(detector2, img_tensor_np, outputs_list, 1))
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()
            outputs1, outputs2 = outputs_list

            # Move outputs to CUDA
            outputs1 = outputs1.cuda()
            outputs2 = outputs2.cuda()

            # Debug raw outputs
            if outputs1.shape[0] > 0:
                logger.debug(f"Model 1 raw scores: {outputs1[:, 4]}")
            if outputs2.shape[0] > 0:
                logger.debug(f"Model 2 raw scores: {outputs2[:, 4]}")

            # Expect (N, 5) shape: [x1, y1, x2, y2, conf]
            logger.debug(f"Model 1 output shape: {outputs1.shape}")
            logger.debug(f"Model 2 output shape: {outputs2.shape}")

            # Visualize raw outputs from Model 1 and Model 2 (in resized image space: 608x1088)
            visualize_xyxy_detections(img, outputs1, frame_id, vis_dir, args.vis_interval, "model1_raw", img_size=img_size)
            visualize_xyxy_detections(img, outputs2, frame_id, vis_dir, args.vis_interval, "model2_raw", img_size=img_size)

            # Run EnsembleDetector
            ensemble_outputs = ensemble_detector(img_np)
            ensemble_outputs = ensemble_outputs.cuda()
            logger.debug(f"Ensemble output shape: {ensemble_outputs.shape}")
            if ensemble_outputs.shape[0] > 0:
                logger.debug(f"Ensemble scores: {ensemble_outputs[:, 4]}")
            # Visualize ensemble outputs (in original image space: 1920x1080)
            visualize_xyxy_detections(img, ensemble_outputs, frame_id, vis_dir, args.vis_interval, "ensemble", img_size=None)

            # Convert to YOLOX format: [cx, cy, w, h, conf, class_score]
            # Since we only have one class (pedestrians), duplicate conf as class_score
            if outputs1.shape[0] > 0:
                outputs1_cxcywh = xyxy2cxcywh(outputs1[:, :4])
                outputs1_yolox = torch.zeros((outputs1.shape[0], 6), dtype=torch.float32, device='cuda')
                outputs1_yolox[:, :4] = outputs1_cxcywh
                outputs1_yolox[:, 4] = outputs1[:, 4]  # conf as objectness score
                outputs1_yolox[:, 5] = outputs1[:, 4]  # conf as class score (single class)
            else:
                outputs1_yolox = torch.zeros((0, 6), dtype=torch.float32, device='cuda')

            if outputs2.shape[0] > 0:
                outputs2_cxcywh = xyxy2cxcywh(outputs2[:, :4])
                outputs2_yolox = torch.zeros((outputs2.shape[0], 6), dtype=torch.float32, device='cuda')
                outputs2_yolox[:, :4] = outputs2_cxcywh
                outputs2_yolox[:, 4] = outputs2[:, 4]  # conf as objectness score
                outputs2_yolox[:, 5] = outputs2[:, 4]  # conf as class score (single class)
            else:
                outputs2_yolox = torch.zeros((0, 6), dtype=torch.float32, device='cuda')

            # Visualize after conversion to YOLOX format (in resized image space: 608x1088)
            visualize_cxcywh_detections(img, outputs1_yolox, frame_id, vis_dir, args.vis_interval, "model1_yolox", img_size=img_size)
            visualize_cxcywh_detections(img, outputs2_yolox, frame_id, vis_dir, args.vis_interval, "model2_yolox", img_size=img_size)

            # Combine outputs and apply weights
            combined = torch.cat((outputs1_yolox, outputs2_yolox), dim=0)
            if combined.shape[0] > 0:
                weights = torch.tensor([args.model1_weight] * outputs1_yolox.shape[0] + 
                                      [args.model2_weight] * outputs2_yolox.shape[0], device='cuda')
                # Only apply weights to scores (columns 4 and 5)
                outputs = combined.clone()
                outputs[:, 4:] = outputs[:, 4:] * weights.view(-1, 1)  # Apply weights to conf and class_score
                logger.debug(f"Combined scores after weighting: {outputs[:, 4]}")
                outputs = outputs.unsqueeze(0)  # Shape: (1, N, 6)
                logger.info(f"Frame {frame_id}: {combined.shape[0]} detections before NMS")
            else:
                outputs = torch.zeros((1, 0, 6), device='cuda')
                logger.info(f"Frame {frame_id}: 0 detections before NMS")

            # Visualize after weighting (in resized image space: 608x1088)
            visualize_cxcywh_detections(img, outputs[0], frame_id, vis_dir, args.vis_interval, "after_weighting", img_size=img_size)

            # Postprocess (expects [cx, cy, w, h, obj_score, class_score])
            outputs = postprocess(outputs, num_classes=1, conf_thre=args.confthre, nms_thre=args.nmsthre)
            if outputs[0] is not None:
                outputs = outputs[0]  # Shape: (N, 7) [x1, y1, x2, y2, obj_score, class_score, class_pred]
                logger.info(f"Frame {frame_id}: {outputs.shape[0]} detections after NMS")
            else:
                outputs = None
                logger.info(f"Frame {frame_id}: 0 detections after NMS")

            # Visualize after postprocessing (in resized image space: 608x1088)
            if outputs is not None:
                visualize_xyxy_detections(img, outputs[:, :6], frame_id, vis_dir, args.vis_interval, "after_postprocess", img_size=img_size)

        if outputs is not None:
            det = outputs[:, :6].cpu().numpy()  # [x1, y1, x2, y2, obj_score, class_score]
            det[:, 4] *= det[:, 5]  # Combine objectness and class score
            det = det[:, :5]  # Keep only [x1, y1, x2, y2, score]
            scale = min(img_size[0] / orig_size[0], img_size[1] / orig_size[1])
            logger.info(f"Final scaling factor: {scale}")
            det[:, :4] /= scale  # Scale back to original resolution
            det_results[video_name][frame_id] = det
        else:
            det_results[video_name][frame_id] = None

        # Visualize final detections (in original image space: 1920x1080)
        visualize_xyxy_detections(img, torch.tensor(det) if det is not None else None, frame_id, vis_dir, args.vis_interval, "final", img_size=None)
        logger.info(f"Processed frame {frame_id} for {video_name}")

    os.makedirs(args.output_folder, exist_ok=True)
    pickle_path = os.path.join(args.output_folder, args.exp_name)
    with open(pickle_path, 'wb') as f:
        pickle.dump(det_results, f)
    logger.info(f"Detections saved to {pickle_path}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)