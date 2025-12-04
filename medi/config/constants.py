# config/constants.py
"""
vis_test.py에서 이식한 상수 및 설정
"""
import numpy as np

# ========================================
# 카메라 캘리브레이션 상수
# ========================================

# 거리 측정을 위한 상수
# 참고: 이 값은 사용자의 웹캠/해상도에 따라 보정(calibration)이 필요합니다.
# 640x480, 60도 화각 웹캠 기준의 추정치
FOCAL_LENGTH_PX = 750 

# 사용자가 가정한 실제 세계의 눈 사이 거리 (cm)
KNOWN_IPD_CM = 6.3

# 다중 얼굴 감지 (타겟 추적용)
MAX_NUM_FACES = 3

# ========================================
# 이벤트 감지 임계값
# ========================================

# 눈 깜박임 측정을 위한 상수
EAR_THRESHOLD = 0.2  # 눈을 감았다고 판단하는 EAR 임계값
BLINK_CONSECUTIVE_FRAMES = 2  # 몇 프레임 이상 눈을 감아야 1회 깜박임으로 인정할지
BLINK_MIN_DURATION_MS = 50  # 최소 깜박임 지속 시간
BLINK_MAX_DURATION_MS = 500  # 최대 깜박임 지속 시간

# 하품 횟수 측정을 위한 상수
MAR_THRESHOLD = 0.65  # 이 비율보다 크면 하품 가능성 높음 (보정 필요)
YAWN_CONSECUTIVE_FRAMES = 10  # 최소 10프레임 (약 0.3~0.5초) 이상 유지되어야 하품으로 인정
YAWN_MIN_DURATION_MS = 300  # 최소 하품 지속 시간

# 끄덕임 감지
NODDING_PITCH_THRESHOLD = 8.0  # 고개 끄덕임으로 판단하는 pitch 변화량 (도)
NODDING_FREQ_MIN = 0.5  # 최소 주파수 (Hz)
NODDING_FREQ_MAX = 2.0  # 최대 주파수 (Hz)

# ========================================
# 품질 관리 임계값
# ========================================

# 조도 (밝기)
BRIGHTNESS_MIN = 30  # 너무 어두움
BRIGHTNESS_MAX = 220  # 너무 밝음 (눈부심)

# 거리
DISTANCE_MIN_CM = 20  # 너무 가까움
DISTANCE_MAX_CM = 120  # 너무 멀음

# 얼굴 각도 (Yaw)
YAW_MAX_ANGLE = 50  # 측면 각도 제한 (도)

# PnP 신뢰도
PNP_MIN_CONFIDENCE = 0.5  # 최소 신뢰도

# ========================================
# MediaPipe 랜드마크 인덱스
# ========================================

class LandmarkIndex:
    """MediaPipe FaceMesh 랜드마크 인덱스 (vis_test.py 기반)"""
    
    # 왼쪽 눈 (EAR 계산용)
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    
    # 오른쪽 눈 (EAR 계산용)
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    
    # 왼쪽 눈동자 (거리 계산용)
    LEFT_IRIS = 473
    
    # 오른쪽 눈동자 (거리 계산용)
    RIGHT_IRIS = 468
    
    # 입술 (MAR 계산용)
    MOUTH_TOP = 13  # 입술 상단 중앙
    MOUTH_BOTTOM = 14  # 입술 하단 중앙
    MOUTH_LEFT = 291  # 입술 왼쪽 끝
    MOUTH_RIGHT = 61  # 입술 오른쪽 끝
    
    # PnP용 6점 (3D 모델 대응)
    NOSE_TIP = 1
    CHIN = 199
    LEFT_EYE_CORNER = 33   # 이미지상 왼쪽 (실제 오른쪽 눈)
    RIGHT_EYE_CORNER = 263  # 이미지상 오른쪽 (실제 왼쪽 눈)
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291

# ========================================
# 3D 얼굴 모델 (PnP용)
# ========================================

# vis_test.py의 GENERIC_3D_MODEL_POINTS
# 일반적인 사람의 얼굴 비율을 딴 3D 좌표입니다. (단위는 임의적이지만 비율이 중요)
# 순서: 코끝, 턱, 왼쪽 눈 끝, 오른쪽 눈 끝, 왼쪽 입가, 오른쪽 입가
GENERIC_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# ========================================
# 타겟 추적 설정
# ========================================

# 히스테리시스 설정
TARGET_LOCK_FRAMES = 30  # 1초 (30fps 기준) - 타겟 전환 시 대기 시간
TARGET_SWITCH_MARGIN_CM = 10.0  # 10cm 차이나야 타겟 전환 고려

# ========================================
# 성능 프로파일
# ========================================

PERFORMANCE_PROFILES = {
    "power_saving": {
        "base_fps": 15,
        "boost_fps": 20,
        "pnp_interval": 10,  # 10프레임마다 PnP 계산
        "brightness_downsample": 4  # 1/4 해상도
    },
    "balanced": {
        "base_fps": 20,
        "boost_fps": 30,
        "pnp_interval": 5,
        "brightness_downsample": 4
    },
    "accuracy": {
        "base_fps": 30,
        "boost_fps": 30,
        "pnp_interval": 1,  # 매 프레임
        "brightness_downsample": 1  # 원본 해상도
    }
}
