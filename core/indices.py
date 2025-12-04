# core/indices.py
def clamp01(x): 
    return max(0.0, min(1.0, float(x)))

def compute_from_features(f):
    # f: dict with keys like perclos, yawn_rate_min, posture_angle_norm, headpose_var, gaze_on_pct, near_work, facial_tension, blink_var
    perclos   = clamp01(f.get("perclos", 0.0))            # 0~1
    yawn      = max(0.0, f.get("yawn_rate_min", 0.0)/6)   # 분당 6회≈1.0
    posture   = clamp01(f.get("posture_angle_norm", 0.0))
    headvar   = clamp01(f.get("headpose_var", 0.0))
    gaze_off  = clamp01(1.0 - f.get("gaze_on_pct", 0.7))
    nearwork  = clamp01(f.get("near_work", 0.0))
    tension   = clamp01(f.get("facial_tension", 0.5))
    blinkvar  = clamp01(f.get("blink_var", 0.2))

    # Fatigue rule (예시 가중치)
    fatigue = (
        0.45*perclos +
        0.20*yawn +
        0.10*blinkvar +
        0.10*nearwork +
        0.15*posture
    ) * 100.0  # 0~100 점수

    # Stress rule (예시 가중치)
    stress = (
        0.30*tension +
        0.20*gaze_off +
        0.20*headvar +
        0.20*posture +
        0.10*nearwork
    ) * 100.0

    return {
        "fatigue": float(max(0.0, min(100.0, fatigue))),
        "stress":  float(max(0.0, min(100.0, stress))),
    }
