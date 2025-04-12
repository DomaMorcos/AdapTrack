import numpy as np
import os
from tqdm import tqdm

def split_gt(input_path, output_dir, seq_name, total_frames, split_ratio=0.8):
    # Read gt.txt
    data = np.loadtxt(input_path, delimiter=',')
    # Filter for class=1 (pedestrian)
    data = data[data[:, 7] == 1]
    
    # Calculate split point
    train_frames = int(total_frames * split_ratio)
    
    # Split into train and val
    train_data = data[data[:, 0] <= train_frames]
    val_data = data[data[:, 0] > train_frames]
    
    # Save train data
    train_path = os.path.join(output_dir, seq_name, 'gt', 'gt_train_half.txt')
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    if len(train_data) > 0:
        np.savetxt(train_path, train_data, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f')
        print(f"Saved {train_path} with {len(train_data)} detections")
    else:
        print(f"Warning: No training data for {seq_name}")
    
    # Save val data
    val_path = os.path.join(output_dir, seq_name, 'gt', 'gt_val_half.txt')
    if len(val_data) > 0:
        np.savetxt(val_path, val_data, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f')
        print(f"Saved {val_path} with {len(val_data)} detections")
    else:
        print(f"Warning: No validation data for {seq_name}")

if __name__ == "__main__":
    root_input = '/kaggle/input/mot20fawry/tracking/train'
    root_output = '/kaggle/working/mot20_split/train'
    sequences = [
        ('02', 2782),
        ('03', 2405),
        ('05', 3315)
    ]

    for seq_name, total_frames in tqdm(sequences, desc="Preprocessing sequences", position=0):
        input_path = os.path.join(root_input, seq_name, 'gt', 'gt.txt')
        split_gt(input_path, root_output, seq_name, total_frames)