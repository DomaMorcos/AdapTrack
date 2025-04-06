import numpy as np
from trackers.kalman_filter import KalmanFilter

class Detection(object):
    def __init__(self, tlbr, confidence, feature):
        self.tlbr = tlbr
        self.tlwh = tlbr.copy()
        self.tlwh[2:] -= self.tlwh[:2]
        self.confidence = confidence
        self.feature = feature

    def to_cxcyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    def __init__(self, cxcyah, track_id, score=None, feature=None, conf_thresh=0.4, min_len=3, ema_beta=0.9, max_age=50):
        self.track_id = track_id
        self.hits = 1
        self.time_since_update = 0
        self.state = TrackState.Tentative

        # Explicit parameters (replacing opt)
        self.conf_thresh = conf_thresh
        self.min_len = min_len
        self.ema_beta = ema_beta
        self.max_age = max_age

        self.scores = []
        if score is not None:
            self.scores.append(score)

        self.features = []
        if feature is not None:
            feature /= np.linalg.norm(feature)
            self.features.append(feature)

        self.kf = KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(cxcyah)

    def predict(self):
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def update(self, detection):
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance,
                                                    detection.to_cxcyah(), detection.confidence)
        feature = detection.feature / np.linalg.norm(detection.feature)
        beta = (detection.confidence - self.conf_thresh) / (1 - self.conf_thresh)
        alpha = self.ema_beta + (1 - self.ema_beta) * (1 - beta)
        smooth_feat = alpha * self.features[-1] + (1 - alpha) * feature
        self.features = [smooth_feat / np.linalg.norm(smooth_feat)]

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self.min_len:
            self.state = TrackState.Confirmed

    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted