# core/calibrator.py
from collections import deque
import statistics as stats

class Calibrator:
    """
    개인 임계 자동화:
      - 워밍업 수집 → baseline 추정
      - 운영 중 품질 양호 프레임에서만 천천히 EWMA 업데이트
      - 눈(EAR) 히스테리시스, 하품(MAR) 임계 제공
    """
    def __init__(
        self,
        warmup_sec=30,
        fps=30,
        ear_scale=0.65,
        ear_open_delta=0.03,
        ewma_alpha=0.02,
        yawn_k=3.0
    ):
        self.warmup_needed = int(warmup_sec * fps)
        self.ear_vals = deque(maxlen=self.warmup_needed)
        self.mar_vals = deque(maxlen=self.warmup_needed)
        self.ready = False

        self.ear_mu = 0.30   # 안전 초기값
        self.ear_scale = ear_scale
        self.ear_open_delta = ear_open_delta
        self.ewma_alpha = ewma_alpha

        self.yawn_k = yawn_k
        self.mar_median = 0.20
        self.mar_mad = 0.03

    def _good_quality(self, q: dict) -> bool:
        if not q: return False
        return (q.get("fps", 0) >= 20) and (q.get("occlusion", 1.0) <= 0.2) and (q.get("lighting", 0) >= 0.4)

    def consume(self, feats: dict):
        """한 프레임의 특징을 받아 워밍업/적응 업데이트"""
        q = feats.get("quality", {})
        ear = feats.get("ear", None)
        mar = feats.get("mar", None)
        if ear is None or mar is None:
            return

        # 워밍업 단계
        if not self.ready:
            if self._good_quality(q):
                self.ear_vals.append(float(ear))
                self.mar_vals.append(float(mar))
            # 60%만 쌓여도 가동
            if len(self.ear_vals) >= int(self.warmup_needed * 0.6):
                self.ear_mu = max(0.15, min(0.45, stats.fmean(self.ear_vals)))
                med = stats.median(self.mar_vals) if self.mar_vals else 0.2
                mad = stats.median([abs(x-med) for x in self.mar_vals]) if self.mar_vals else 0.03
                self.mar_median, self.mar_mad = med, max(mad, 1e-3)
                self.ready = True
            return

        # 운영 중: 좋은 품질 프레임에서만 느리게 EWMA
        if self._good_quality(q):
            a = self.ewma_alpha
            self.ear_mu = (1 - a) * self.ear_mu + a * float(ear)
            # MAR도 천천히 갱신(큰 변동은 제외)
            med = self.mar_median
            if abs(mar - med) < 0.2:
                self.mar_median = 0.9 * self.mar_median + 0.1 * float(mar)

    @property
    def th_close(self) -> float:
        return self.ear_mu * self.ear_scale

    @property
    def th_open(self) -> float:
        return self.th_close + self.ear_open_delta

    @property
    def th_yawn(self) -> float:
        return self.mar_median + self.yawn_k * self.mar_mad

    def get_progress(self) -> float:
        """캘리브레이션 진행률 (0-100%)"""
        if self.ready:
            return 100.0
        required = int(self.warmup_needed * 0.6)
        current = len(self.ear_vals)
        return min(100.0, (current / required) * 100.0) if required > 0 else 0.0
