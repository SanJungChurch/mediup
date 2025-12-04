
import numpy as np
import time

try:
    import mediapipe as mp
    _HAS_MP = True
except Exception:
    _HAS_MP = False
    mp = None

class FaceMeshWrapper:
    """
    MediaPipe FaceMesh + (옵션) Pose 래퍼.
    vis_test.py 기반 타겟 추적 기능 추가
    
    process(frame[BGR]) -> dict:
        {
          "face_landmarks": [(x,y,z), ...]  # 정규화(0..1) 좌표
          "pose_landmarks": [(x,y,z,vis), ...] or None
          "image_shape": (h, w)
          "target_face_idx": int or None  # 타겟 얼굴 인덱스
          "all_faces": [[(x,y,z), ...], ...]  # 모든 감지된 얼굴
        }
    """
    def __init__(self, use_pose=True, max_num_faces=1, use_target_tracking=False):
        self.use_pose = use_pose and _HAS_MP
        self.use_target_tracking = use_target_tracking
        self.max_num_faces = max_num_faces if use_target_tracking else 1
        
        # 타겟 추적 상태
        self.current_target_idx = None
        self.target_lock_frames = 0
        self.LOCK_THRESHOLD = 30  # 1초 (30fps 기준)
        self.SWITCH_MARGIN = 10.0  # 10cm 차이나야 전환
        
        if _HAS_MP:
            self.mp_face = mp.solutions.face_mesh
            self.mp_pose = mp.solutions.pose
            self.face = self.mp_face.FaceMesh(
                max_num_faces=self.max_num_faces,
                refine_landmarks=True,
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) if self.use_pose else None
        else:
            self.face = None
            self.pose = None

    def close(self):
        if self.face: self.face.close()
        if self.pose: self.pose.close()

    def _calculate_distance(self, landmarks, img_width, img_height):
        """
        vis_test.py의 display2face_dist() 이식
        눈동자 사이 거리(IPD)로 모니터-얼굴 거리 추정
        """
        try:
            # 좌우 눈동자 (iris) 랜드마크
            LEFT_IRIS = 473
            RIGHT_IRIS = 468
            
            left_iris = landmarks[LEFT_IRIS]
            right_iris = landmarks[RIGHT_IRIS]
            
            # 픽셀 좌표 변환
            p1 = (int(left_iris.x * img_width), int(left_iris.y * img_height))
            p2 = (int(right_iris.x * img_width), int(right_iris.y * img_height))
            
            # 픽셀 거리
            import math
            ipd_px = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            
            # cm로 변환 (삼각법)
            FOCAL_LENGTH_PX = 750
            KNOWN_IPD_CM = 6.3
            
            if ipd_px == 0:
                return float('inf')
            
            distance_cm = (KNOWN_IPD_CM * FOCAL_LENGTH_PX) / ipd_px
            return distance_cm
        except Exception:
            return float('inf')

    def _get_target_face_index(self, multi_face_landmarks, img_height, img_width):
        """
        vis_test.py의 get_target_face_index() 이식
        다중 얼굴 중 가장 가까운 얼굴을 타겟으로 선정 (히스테리시스 적용)
        """
        if not multi_face_landmarks:
            self.current_target_idx = None
            self.target_lock_frames = 0
            return None, None
        
        # 모든 얼굴의 거리 계산
        distances = []
        for landmarks in multi_face_landmarks:
            dist = self._calculate_distance(landmarks.landmark, img_width, img_height)
            distances.append(dist)
        
        # 초기 상태: 가장 가까운 얼굴 선택
        if self.current_target_idx is None:
            self.current_target_idx = int(np.argmin(distances))
            self.target_lock_frames = 0
            return self.current_target_idx, distances[self.current_target_idx]
        
        # 현재 타겟이 여전히 존재하는지 확인
        if self.current_target_idx >= len(distances):
            self.current_target_idx = int(np.argmin(distances))
            self.target_lock_frames = 0
            return self.current_target_idx, distances[self.current_target_idx]
        
        current_dist = distances[self.current_target_idx]
        closest_idx = int(np.argmin(distances))
        closest_dist = distances[closest_idx]
        
        # 다른 얼굴이 현재 타겟보다 SWITCH_MARGIN 이상 가까워야 전환 고려
        if closest_idx != self.current_target_idx:
            if current_dist - closest_dist > self.SWITCH_MARGIN:
                self.target_lock_frames += 1
                
                # LOCK_THRESHOLD 프레임 이상 유지되면 전환
                if self.target_lock_frames >= self.LOCK_THRESHOLD:
                    self.current_target_idx = closest_idx
                    self.target_lock_frames = 0
                    print(f"⚠️ 타겟 전환: Face #{self.current_target_idx}")
            else:
                self.target_lock_frames = 0  # 리셋
        else:
            self.target_lock_frames = 0
        
        return self.current_target_idx, current_dist

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        result = {
            "face_landmarks": None, 
            "pose_landmarks": None, 
            "image_shape": (h, w),
            "target_face_idx": None,
            "all_faces": [],
            "target_distance_cm": None
        }
        
        if not _HAS_MP:
            return result

        # MediaPipe는 RGB 입력
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        f = self.face.process(rgb)

        if f and f.multi_face_landmarks:
            # 모든 얼굴 저장
            all_faces = []
            for face in f.multi_face_landmarks:
                pts = [(lm.x, lm.y, getattr(lm, "z", 0.0)) for lm in face.landmark]
                all_faces.append(pts)
            result["all_faces"] = all_faces
            
            # 타겟 추적 사용 시
            if self.use_target_tracking and len(f.multi_face_landmarks) > 1:
                target_idx, target_dist = self._get_target_face_index(
                    f.multi_face_landmarks, h, w
                )
                result["target_face_idx"] = target_idx
                result["target_distance_cm"] = target_dist
                
                if target_idx is not None:
                    result["face_landmarks"] = all_faces[target_idx]
            else:
                # 단일 얼굴 또는 타겟 추적 비활성화
                result["face_landmarks"] = all_faces[0]
                result["target_face_idx"] = 0

        if self.pose:
            p = self.pose.process(rgb)
            if p and p.pose_landmarks:
                pls = []
                for lm in p.pose_landmarks.landmark:
                    pls.append((lm.x, lm.y, lm.z, lm.visibility))
                result["pose_landmarks"] = pls

        return result
