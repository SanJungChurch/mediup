
import numpy as np
from datetime import datetime
from collections import defaultdict

class GraphAnalyzer:
    def __init__(self):
        # 분석 대상 컬럼 6개
        self.targets = [
            "perclos", "yawn_rate", "posture_angle", 
            "headpose_var", "fatigue", "stress"
        ]

    def analyze(self, raw_data: list, trend_window_min: int = 10):
        """
        1. Hourly Averages (시간대별 평균)
        2. Recent Trend Slope (최근 N분간 분당 변화율)
        """
        if not raw_data:
            return "데이터가 충분하지 않습니다."

        # 데이터 전처리: timestamp 문자열을 datetime 객체로 변환
        processed = []
        for row in raw_data:
            dt = datetime.fromisoformat(row['ts'])
            processed.append({**row, 'dt': dt, 'ts_unix': dt.timestamp()})

        # 결과 저장소
        report = ["[📊 시간대별 평균 및 트렌드 분석]"]

        # === 1. Hourly Average (시간대별 평균) ===
        hourly_groups = defaultdict(list)
        for row in processed:
            hour_key = row['dt'].strftime("%H시") # 예: "14시"
            hourly_groups[hour_key].append(row)

        report.append("\n1️⃣ 시간대별 평균 (Hourly Avg):")
        
        # 정렬된 시간 순서대로 출력
        sorted_hours = sorted(hourly_groups.keys())
        for hour in sorted_hours:
            rows = hourly_groups[hour]
            # 각 지표별 평균 계산
            stats = []
            for key in ["fatigue", "stress", "perclos", "yawn_rate", "posture_angle", "headpose_var"]:
                vals = [r[key] for r in rows if r[key] is not None]
                if vals:
                    avg = sum(vals) / len(vals)
                    stats.append(f"{key}:{avg:.1f}")
            report.append(f" - {hour}: {', '.join(stats)}")

        # === 2. Recent Trend (선형 회귀 기울기) ===
        # 최근 N분 데이터 필터링
        now_ts = processed[-1]['ts_unix']
        start_ts = now_ts - (trend_window_min * 60)
        
        recent_data = [r for r in processed if r['ts_unix'] >= start_ts]

        report.append(f"\n2️⃣ 최근 {trend_window_min}분 트렌드 (분당 변화율):")
        
        if len(recent_data) < 10: # 데이터가 너무 적으면 분석 불가
            report.append(" - (분석을 위한 데이터가 모이는 중입니다)")
        else:
            # X축: 시간 (분 단위로 정규화, 0분 ~ N분)
            # Y축: 각 지표 값
            x = np.array([r['ts_unix'] for r in recent_data])
            x = (x - x.min()) / 60.0  # 초 단위를 '분' 단위로 변환 (Slope 의미 명확화)

            for target in self.targets:
                y = np.array([r[target] for r in recent_data])
                
                # 1차 함수(선형 회귀) 적합: y = slope * x + intercept
                # polyfit(deg=1)의 첫 번째 반환값이 기울기(slope)
                if len(y) > 0:
                    slope, _ = np.polyfit(x, y, 1)
                    
                    # LLM이 이해하기 쉬운 텍스트로 변환
                    # 기울기가 0.0에 가까우면 '유지', 양수면 '증가', 음수면 '감소'
                    # 하지만 우리는 "판단"하지 않고 "값"을 줍니다.
                    direction = "↗️증가" if slope > 0 else "↘️감소"
                    if abs(slope) < 0.01: direction = "➡️유지"
                    
                    report.append(f" - {target}: {direction} (속도: {slope:+.3f}/분)")

        return "\n".join(report)