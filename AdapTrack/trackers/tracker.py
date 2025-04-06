from trackers.cmc import *
from trackers import metrics
from trackers.units import Track
from trackers import linear_assignment
import numpy as np

class Tracker:
    def __init__(self, metric, vid_name, max_age=50, max_distance=0.45, max_iou_distance=0.70, conf_thresh=0.4, min_len=3, ema_beta=0.9):
        self.metric = metric
        self.max_age = max_age
        self.max_distance = max_distance
        self.max_iou_distance = max_iou_distance
        self.conf_thresh = conf_thresh
        self.min_len = min_len
        self.ema_beta = ema_beta

        self.tracks = []
        self.next_id = 1
        self.cmc = CMC(vid_name)

    def initiate_track(self, detection):
        self.tracks.append(Track(detection.to_cxcyah(), self.next_id, detection.confidence, detection.feature,
                                 conf_thresh=self.conf_thresh, min_len=self.min_len, ema_beta=self.ema_beta, max_age=self.max_age))
        self.next_id += 1

    def predict(self):
        for track in self.tracks:
            track.predict()

    def camera_update(self):
        warp_matrix = self.cmc.get_warp_matrix()
        for track in self.tracks:
            apply_cmc(track, warp_matrix)

    def gated_metric(self, tracks, detections, track_indices, detection_indices):
        targets = np.array([tracks[i].track_id for i in track_indices])
        features = np.array([detections[i].feature for i in detection_indices])
        cost_matrix = self.metric.distance(features, targets)
        cost_matrix_min = np.min(cost_matrix)
        cost_matrix_max = np.max(cost_matrix)
        cost_matrix = linear_assignment.gate_cost_matrix(cost_matrix, tracks, detections, track_indices, detection_indices)
        return cost_matrix, cost_matrix_min, cost_matrix_max

    def match(self, detections):
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        matches_a, _, unmatched_detections = \
            linear_assignment.min_cost_matching([self.gated_metric, metrics.iou_constraint, True],
                                                self.max_distance, self.tracks, detections, confirmed_tracks)
        unmatched_tracks_a = list(set(confirmed_tracks) - set(k for k, _ in matches_a))
        candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching([metrics.iou_cost, None, True], self.max_iou_distance, self.tracks,
                                                detections, candidates, unmatched_detections)
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections):
        matches, unmatched_tracks, unmatched_detections = self.match(detections)
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            if detections[detection_idx].confidence >= self.conf_thresh:
                self.initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)