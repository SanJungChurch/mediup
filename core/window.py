
from collections import deque
import time, statistics as stats

class WindowAggregator:
    def __init__(self, window_sec: int = 60):
        self.window_ms = window_sec * 1000
        self.samples = deque()   # (ts_ms, feats, events)
        self._last_cleanup = 0

    def update(self, feats: dict, events: dict):
        ts = int(time.time()*1000)
        self.samples.append((ts, feats, events))
        # cleanup
        if ts - self._last_cleanup > 2000:
            self._cleanup(ts)
            self._last_cleanup = ts

    def _cleanup(self, now_ms):
        while self.samples and (now_ms - self.samples[0][0]) > self.window_ms:
            self.samples.popleft()

    def snapshot(self):
        if not self.samples:
            return {}
        # 집계
        perclos = self._perclos()
        blink_rate = self._rate_per_min("blink")
        yawn_rate = self._rate_per_min("yawn")
        # 간단 평균들
        posture = self._avg_feat("posture_angle_norm", 0.0)
        headvar = self._var_feat("ear")  # 대용(추후 head pose 분산)
        gaze_on = self._avg_feat("gaze_on_pct", 0.7)
        near = self._avg_feat("near_work", 0.0)

        snap = {
            "perclos": perclos,
            "blink_rate_min": blink_rate,
            "yawn_rate_min": yawn_rate,
            "posture_angle_norm": posture,
            "headpose_var": headvar,
            "gaze_on_pct": gaze_on,
            "near_work": near,
        }
        return snap

    def _perclos(self, th=0.21):
        cnt = 0; closed = 0
        for _, feats, _ in self.samples:
            cnt += 1
            if feats.get("ear", 0.3) < th:
                closed += 1
        return (closed/cnt) if cnt>0 else 0.0

    def _rate_per_min(self, key):
        if not self.samples: return 0.0
        # 이벤트 카운트
        ev_count = sum( e.get(key,0) for _,_,e in self.samples )
        dur_ms = self.samples[-1][0] - self.samples[0][0] + 1
        minutes = max(1e-3, dur_ms/60000.0)
        return ev_count / minutes

    def _avg_feat(self, name, default=0.0):
        vals = [ f.get(name, default) for _,f,_ in self.samples ]
        return sum(vals)/len(vals) if vals else default

    def _var_feat(self, name, default=0.0):
        vals = [ f.get(name, default) for _,f,_ in self.samples ]
        return (stats.pvariance(vals) if len(vals)>=2 else 0.0)

class Calibrator:
    """초기 30~60초 개인 기준선/임계 계산 자리 (간단 스텁)."""
    def __init__(self):
        self.ready = True
        self.ear_base = 0.3
        self.th_close_scale = 0.65
        self.th_open_delta = 0.03

    def consume(self, feats: dict):
        # TODO: ear 이동평균으로 ear_base 업데이트
        pass

    @property
    def th_close(self):
        return self.ear_base * self.th_close_scale

    @property
    def th_open(self):
        return self.th_close + self.th_open_delta
