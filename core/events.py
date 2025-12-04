
from collections import deque
import time

class EventState:
    """
    Blink / Yawn / Nodding 이벤트를 히스테리시스로 검출.
    - blink: EAR < th_close 지속 후 th_open 회복 시 1회
    - yawn : MAR > th_yawn 최소 ms 지속 시 1회
    - nod  : pitch 주기 분석 대신 간단 지속 카운트(후속 개선)
    """
    def __init__(self, fps=30, th_close=0.21, th_open=0.25, th_yawn=0.60, yawn_min_ms=800):
        self.fps = fps
        self.th_close = th_close
        self.th_open = th_open
        self.th_yawn = th_yawn
        self.yawn_min_ms = yawn_min_ms

        self._eye_closed = False
        self._eye_ts = 0.0

        self._yawn_on = False
        self._yawn_start = 0.0

        self._blink_times = deque(maxlen=120)  # 최근 2분

    def update(self, feats):
        now = time.time()*1000.0
        ear = feats.get("ear", 0.3)
        mar = feats.get("mar", 0.2)

        blink = 0
        if not self._eye_closed and ear < self.th_close:
            self._eye_closed = True
            self._eye_ts = now
        elif self._eye_closed and ear > self.th_open:
            self._eye_closed = False
            blink = 1
            self._blink_times.append(now)

        yawn = 0
        if not self._yawn_on and mar > self.th_yawn:
            self._yawn_on = True
            self._yawn_start = now
        elif self._yawn_on and mar <= self.th_yawn:
            dur = now - self._yawn_start
            self._yawn_on = False
            if dur >= self.yawn_min_ms:
                yawn = 1

        # nodding은 후속 개선. 일단 0
        nod = 0

        # 분당 깜빡임/하품 비율 계산은 window에서 처리
        return {"blink": blink, "yawn": yawn, "nodding": nod}
