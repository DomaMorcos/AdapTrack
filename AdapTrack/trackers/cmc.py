import numpy as np
import os

class CMC:
    def __init__(self, vid_name):
        super(CMC, self).__init__()
        # Use absolute path instead of relative
        base_dir = "/kaggle/working/AdapTrack/AdapTrack"  # Adjust if your base dir differs
        self.gmc_path = os.path.join(base_dir, "trackers", "cmc", f"GMC-{vid_name}.txt")
        print(f"Looking for GMC file at: {self.gmc_path}")  # Debug path
        try:
            self.gmcFile = open(self.gmc_path, 'r')
            self.frame_map = {}
            for line in self.gmcFile:
                tokens = line.strip().split("\t")
                if len(tokens) >= 7:
                    try:
                        frame_id = int(tokens[0])
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                        warp_matrix[0, 0] = float(tokens[1])
                        warp_matrix[0, 1] = float(tokens[2])
                        warp_matrix[0, 2] = float(tokens[3])
                        warp_matrix[1, 0] = float(tokens[4])
                        warp_matrix[1, 1] = float(tokens[5])
                        warp_matrix[1, 2] = float(tokens[6])
                        self.frame_map[frame_id] = warp_matrix
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Skipping malformed line in GMC file: '{line.strip()}': {e}")
            self.gmcFile.close()
            print(f"Loaded {len(self.frame_map)} GMC entries for {vid_name}")
        except FileNotFoundError:
            print(f"Warning: GMC file {self.gmc_path} not found. Using identity matrix for all frames.")
            self.gmcFile = None
            self.frame_map = {}

    def get_warp_matrix(self, frame_id=None):
        if self.gmcFile is None or not self.frame_map:
            return np.eye(2, 3, dtype=np.float32)
        return self.frame_map.get(frame_id, np.eye(2, 3, dtype=np.float32))

    def __del__(self):
        if self.gmcFile is not None and not self.gmcFile.closed:
            self.gmcFile.close()

def apply_cmc(track, warp_matrix=np.eye(2, 3)):
    x1, y1, x2, y2 = track.to_tlbr()
    x1_, y1_ = warp_matrix @ np.array([x1, y1, 1]).T
    x2_, y2_ = warp_matrix @ np.array([x2, y2, 1]).T
    w, h = x2_ - x1_, y2_ - y1_
    cx, cy = x1_ + w / 2, y1_ + h / 2
    track.mean[:4] = [cx, cy, w / h, h]