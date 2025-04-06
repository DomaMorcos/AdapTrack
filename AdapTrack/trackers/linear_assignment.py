import numpy as np
from sklearn.mixture import GaussianMixture
import trackers.kalman_filter as kalman_filter
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

def gate_cost_matrix(cost_matrix, tracks, detections, track_indices, detection_indices, gating_lambda=0.98):
    gating_threshold = kalman_filter.chi2inv95[4]
    measurements = np.asarray([detections[i].to_cxcyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = track.kf.gating_distance(track.mean, track.covariance, measurements)
        cost_matrix[row, gating_distance > gating_threshold] = 1e5
        cost_matrix[row] = gating_lambda * cost_matrix[row] + (1 - gating_lambda) * gating_distance
    return cost_matrix

def set_threshold(dists, ori_threshold, min_anchor, max_anchor):
    threshold = ori_threshold
    dists_1d = dists.reshape(-1, 1)
    dists_1d = dists_1d[dists_1d < max_anchor]
    dists_1d = dists_1d[min_anchor < dists_1d]
    if len(dists_1d) > 0:
        dists_1d = list(dists_1d) + [min_anchor, max_anchor]
        dists_1d = np.array(dists_1d).reshape(-1, 1)
        model = KMeans(n_clusters=2, init=np.array([[min_anchor], [max_anchor]]), n_init=1, random_state=10000)
        result = model.fit_predict(dists_1d)
        if np.sum(result == 0) == 0 or np.sum(result == 1) == 0:
            return ori_threshold
        threshold = min(np.max(dists_1d[result == 0]), np.max(dists_1d[result == 1])) + 1e-5
    return threshold

def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices

    distance_metric, constraint_metric, adap_flag = distance_metric
    cost_matrix, _, cost_matrix_max = distance_metric(tracks, detections, track_indices, detection_indices)
    if constraint_metric is not None:
        constraint_matrix = constraint_metric(tracks, detections, track_indices, detection_indices)
        cost_matrix[constraint_matrix == 1] = 1
    if adap_flag:
        max_distance = set_threshold(cost_matrix, max_distance, 0., cost_matrix_max)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    indices = linear_assignment(cost_matrix)
    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[0]:
            unmatched_tracks.append(track_idx)
    for row, col in np.concatenate([indices[0][:, None], indices[1][:, None]], axis=1):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections