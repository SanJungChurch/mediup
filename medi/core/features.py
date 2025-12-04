
import math
import numpy as np

def _dist(p1, p2, w, h):
    x1,y1 = p1[0]*w, p1[1]*h
    x2,y2 = p2[0]*w, p2[1]*h
    return math.hypot(x1-x2, y1-y2)

def _eye_ear(landmarks, w, h, idxs):
    # idxs: [p1,p2,p3,p4,p5,p6] = [left, top1, top2, right, bot1, bot2] (MediaPipe FaceMesh 기준)
    p1,p2,p3,p4,p5,p6 = [landmarks[i] for i in idxs]
    # EAR ~ (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    num = _dist(p2,p6,w,h) + _dist(p3,p5,w,h)
    den = 2.0 * _dist(p1,p4,w,h)
    return (num/den) if den>1e-6 else 0.0

def _mouth_mar(landmarks, w, h):
    # 간단한 MAR: 수직(13-14) / 수평(78-308)
    try:
        v = _dist(landmarks[13], landmarks[14], w, h)
        hlen = _dist(landmarks[78], landmarks[308], w, h)
        return (v / hlen) if hlen>1e-6 else 0.0
    except Exception:
        return 0.0

def _interpupil(landmarks, w, h):
    # 좌우 눈꼬리(33,263) 거리
    try:
        return _dist(landmarks[33], landmarks[263], w, h)
    except Exception:
        return None

def _roll_pitch_proxy(landmarks):
    # 간단 프록시: 좌우 눈 y 차(roll), 코(1)와 눈중심의 y 차(pitch)로 근사
    # 반환은 라디안 값 근사
    try:
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose = landmarks[1]
        cy = (left_eye[1] + right_eye[1]) / 2.0
        roll = math.atan2((right_eye[1]-left_eye[1]), (right_eye[0]-left_eye[0]+1e-6))
        pitch = math.atan2((nose[1]-cy), 0.5)  # 스케일 임의
        return roll, pitch
    except Exception:
        return 0.0, 0.0

# ========================================
# vis_test.py 이식 함수들
# ========================================

def calculate_head_pose_pnp(landmarks, img_width, img_height):
    """
    vis_test.py의 get_head_pose() 이식
    PnP 알고리즘으로 정확한 pitch/yaw/roll 계산
    
    Returns:
        dict: {
            "pitch": float,
            "yaw": float,
            "roll": float,
            "confidence": float,  # 0-1
            "method": "pnp"
        }
        or None if failed
    """
    try:
        import cv2
        
        # 3D 모델 포인트 (vis_test GENERIC_3D_MODEL_POINTS)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # 대응하는 2D 랜드마크 인덱스
        NOSE_TIP = 1
        CHIN = 199
        LEFT_EYE_CORNER = 33
        RIGHT_EYE_CORNER = 263
        LEFT_MOUTH_CORNER = 61
        RIGHT_MOUTH_CORNER = 291
        
        # 2D 이미지 좌표 추출
        image_points = np.array([
            (landmarks[NOSE_TIP][0] * img_width, landmarks[NOSE_TIP][1] * img_height),
            (landmarks[CHIN][0] * img_width, landmarks[CHIN][1] * img_height),
            (landmarks[LEFT_EYE_CORNER][0] * img_width, landmarks[LEFT_EYE_CORNER][1] * img_height),
            (landmarks[RIGHT_EYE_CORNER][0] * img_width, landmarks[RIGHT_EYE_CORNER][1] * img_height),
            (landmarks[LEFT_MOUTH_CORNER][0] * img_width, landmarks[LEFT_MOUTH_CORNER][1] * img_height),
            (landmarks[RIGHT_MOUTH_CORNER][0] * img_width, landmarks[RIGHT_MOUTH_CORNER][1] * img_height)
        ], dtype=np.float64)
        
        # 카메라 매트릭스 생성 (내부 파라미터)
        focal_length = img_width
        center = (img_width / 2, img_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 왜곡 계수 (없다고 가정)
        dist_coeffs = np.zeros((4, 1))
        
        # PnP 문제 풀기
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return None
        
        # 회전 벡터를 회전 행렬로 변환
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # 오일러 각도 추출 (pitch, yaw, roll)
        # pitch: 고개 숙임(+) / 젖힘(-)
        # yaw: 왼쪽 봄(+) / 오른쪽 봄(-)
        # roll: 왼쪽으로 기울임(-) / 오른쪽으로 기울임(+)
        pitch = -math.asin(rotation_matrix[2][0]) * (180.0 / math.pi)
        yaw = math.atan2(rotation_matrix[2][1], rotation_matrix[2][2]) * (180.0 / math.pi)
        roll = math.atan2(rotation_matrix[1][0], rotation_matrix[0][0]) * (180.0 / math.pi)
        
        # 재투영 오차로 신뢰도 계산
        projected_points, _ = cv2.projectPoints(
            model_points,
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeffs
        )
        
        reprojection_error = np.mean(np.linalg.norm(
            projected_points.reshape(-1, 2) - image_points,
            axis=1
        ))
        
        # 신뢰도: 오차가 작을수록 높음 (0-1)
        confidence = 1.0 / (1.0 + reprojection_error / 10.0)
        
        return {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
            "confidence": confidence,
            "method": "pnp"
        }
        
    except Exception as e:
        print(f"PnP HeadPose 계산 실패: {e}")
        return None

def measure_brightness_hsv(image):
    """
    vis_test.py의 measure_brightness() 이식
    HSV 변환 후 V 채널 평균값으로 조도 측정
    
    Args:
        image: BGR 이미지
    
    Returns:
        float: 밝기 값 (0-255)
    """
    try:
        import cv2
        
        # BGR → HSV 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # V 채널 (밝기) 추출
        v_channel = hsv[:, :, 2]
        
        # 평균값 계산
        brightness = float(np.mean(v_channel))
        
        return brightness
        
    except Exception as e:
        print(f"조도 측정 실패: {e}")
        return 128.0  # 중간값

def calculate_neck_angle(pitch, roll):
    """
    거북목(FHP - Forward Head Posture) 각도 계산
    
    Args:
        pitch: 고개 숙임 각도 (도)
        roll: 좌우 기울임 각도 (도)
    
    Returns:
        dict: {
            "fhp_angle": float,  # 0-1 정규화
            "severity": str  # "normal" / "mild" / "moderate" / "severe"
        }
    """
    # 절댓값 (방향 무관)
    abs_pitch = abs(pitch)
    abs_roll = abs(roll)
    
    # 거북목 판정 (임상 기준)
    if abs_pitch < 15:
        severity = "normal"
        fhp_angle = abs_pitch / 40.0  # 0-1 정규화 (40도 기준)
    elif abs_pitch < 25:
        severity = "mild"
        fhp_angle = abs_pitch / 40.0
    elif abs_pitch < 35:
        severity = "moderate"
        fhp_angle = abs_pitch / 40.0
    else:
        severity = "severe"
        fhp_angle = min(1.0, abs_pitch / 40.0)
    
    # Roll도 고려 (목 비틀림)
    if abs_roll > 15:
        fhp_angle = min(1.0, fhp_angle + abs_roll / 60.0)
        if severity == "normal":
            severity = "mild"
    
    return {
        "fhp_angle": fhp_angle,
        "severity": severity
    }

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE= [263, 387, 385, 362, 380, 373]

def compute_all(frame_bgr, lm_dict, use_pnp=False, use_brightness=False):
    """
    입력:
      frame_bgr: BGR 이미지
      lm_dict: facemesh.process() 결과(dict)
      use_pnp: PnP 알고리즘 사용 여부
      use_brightness: 조도 측정 여부
      
    출력(dict):
      perclos, yawn_rate_min(즉시 0), posture_angle_norm, headpose_var(0),
      gaze_on_pct(추정치), distance_cm(추정치), near_work(0/1), quality(dict)
      + head_pose(dict), brightness(float), fhp_info(dict)
    """
    h, w = lm_dict.get("image_shape", (None, None))
    face_lms = lm_dict.get("face_landmarks")
    pose_lms = lm_dict.get("pose_landmarks")
    target_distance = lm_dict.get("target_distance_cm")

    perclos = 0.0
    mar = 0.0
    ipd = None
    roll = pitch = yaw = 0.0
    head_pose_dict = None
    fhp_info = None

    if face_lms:
        ear_l = _eye_ear(face_lms, w, h, LEFT_EYE)
        ear_r = _eye_ear(face_lms, w, h, RIGHT_EYE)
        ear = (ear_l + ear_r) / 2.0
        
        mar = _mouth_mar(face_lms, w, h)
        ipd = _interpupil(face_lms, w, h)
        
        # HeadPose 계산
        if use_pnp:
            head_pose_dict = calculate_head_pose_pnp(face_lms, w, h)
            if head_pose_dict:
                pitch = head_pose_dict["pitch"]
                yaw = head_pose_dict["yaw"]
                roll = head_pose_dict["roll"]
                
                # 거북목 계산
                fhp_info = calculate_neck_angle(pitch, roll)
        
        # Fallback: 프록시 방식
        if head_pose_dict is None:
            roll, pitch = _roll_pitch_proxy(face_lms)
            head_pose_dict = {
                "pitch": pitch * (180.0 / math.pi),
                "yaw": 0.0,
                "roll": roll * (180.0 / math.pi),
                "confidence": 0.5,
                "method": "proxy"
            }
            fhp_info = {"fhp_angle": 0.5, "severity": "unknown"}
    else:
        ear = 0.3  # 안전 기본값

    # 조도 측정
    brightness = 128.0
    if use_brightness and frame_bgr is not None:
        brightness = measure_brightness_hsv(frame_bgr)

    # 품질 지표
    lighting_quality = "good"
    if brightness < 30:
        lighting_quality = "too_dark"
    elif brightness < 80:
        lighting_quality = "dim"
    elif brightness > 220:
        lighting_quality = "too_bright"
    elif brightness > 180:
        lighting_quality = "bright"
    
    quality = {
        "lighting": brightness / 255.0,
        "lighting_quality": lighting_quality,
        "fps": 30.0,  # 실제 FPS는 외부에서 측정
        "occlusion": 0.0 if face_lms else 1.0,
        "target_locked": lm_dict.get("target_face_idx") is not None
    }

    # 거북목/자세
    posture_angle_norm = fhp_info["fhp_angle"] if fhp_info else min(1.0, abs(pitch) / 40.0)

    # 응시율/거리
    gaze_on_pct = 0.7
    
    # 거리 추정
    if target_distance is not None:
        distance_cm = target_distance
    elif ipd and ipd > 1e-3:
        distance_cm = (6000.0 / ipd)
    else:
        distance_cm = 50.0
    
    near_work = 1.0 if distance_cm <= 40.0 else 0.0

    feats = {
        "ear": ear,
        "mar": mar,
        "perclos": 0.0,             # 윈도우 집계에서 대체
        "yawn_rate_min": 0.0,       # 이벤트에서 대체
        "nodding_rate_min": 0.0,    # 이벤트에서 대체
        "posture_angle_norm": posture_angle_norm,
        "headpose_var": 0.0,        # 윈도우에서 분산 계산 가능
        "gaze_on_pct": gaze_on_pct,
        "distance_cm": distance_cm,
        "near_work": near_work,
        "facial_tension": 0.5,      # TODO: 미세변동 기반 추정
        "blink_var": 0.2,           # TODO: 간격 변이
        "quality": quality,
        # vis_test 추가 피처
        "head_pose": head_pose_dict,
        "brightness": brightness,
        "fhp_info": fhp_info
    }
    return feats
