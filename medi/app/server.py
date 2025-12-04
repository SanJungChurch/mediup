# app/server.py
import logging
logging.basicConfig(level=logging.INFO)

import asyncio, collections, time, json, base64, cv2, numpy as np
import yaml
from datetime import datetime, timezone
from typing import Dict
from pathlib import Path

from fastapi import FastAPI, WebSocket, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketDisconnect

from core.capture import Camera
from core.facemesh import FaceMeshWrapper
from core.features import compute_all
from core.events import EventState
from core.window import WindowAggregator
from core.calibrator import Calibrator
from core.indices import compute_from_features

# ⬇ LLM (로컬 우선 / 최초 1회만 HF) + 디버그 상태
from llm.exaone import build_coaching_text, exaone_debug_status

# ⬇ DB 및 트렌드 분석 (v0.8.0)
from db.repository import repo
from core.trend_analysis import GraphAnalyzer

# 로깅 인터벌
LOG_INTERVAL = 10.0  # 10초마다 DB 저장

# ========================================
# 설정 파일 로드
# ========================================
def load_config():
    """config/app.yaml 로드"""
    config_path = Path(__file__).parent.parent / "config" / "app.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

app = FastAPI(title="RuleVision")

# 정적파일
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def index():
    return FileResponse("app/static/index.html")

@app.get("/favicon.ico")
def favicon():
    # 있으면 제공, 없으면 404 대신 빈 응답 방지용
    try:
        return FileResponse("app/static/favicon.ico")
    except Exception:
        return FileResponse("app/static/index.html")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}

@app.get("/llm/health")
def llm_health():
    """
    LLM 실제 호출을 통해 로더/디바이스/폴백 여부를 점검.
    exaone_debug_status()는 내부 상태(로드여부/로컬경로/최근오류 등) 반환.
    """
    demo_stats = {"avg_fatigue": 10.0, "avg_stress": 20.0, "perclos": 0.12}
    demo_docs  = [{"title": "health-check", "path": "local"}]
    txt = build_coaching_text(demo_stats, demo_docs)
    used_fallback = txt.startswith("[LLM:fallback]")
    preview = txt.replace("[LLM:local]\n","").replace("[LLM:fallback]\n","")[:180]
    return {
        "ok": True,
        "used_fallback": used_fallback,
        "preview": preview,
        "exaone": exaone_debug_status(),
    }

# ---- FaceMesh 시각화용 주요 인덱스 ----
LEFT_EYE_RING  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263]
MOUTH_OUTER    = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61]
MOUTH_INNER    = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]

def _pt(img, lm, i):
    h, w = img.shape[:2]
    x = int(lm[i][0] * w)
    y = int(lm[i][1] * h)
    return (x, y)

def _polyline(img, lm, idxs, color, closed=False, thickness=1):
    pts = [ _pt(img, lm, i) for i in idxs if i < len(lm) ]
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], closed, color, thickness, cv2.LINE_AA)

def draw_debug_overlay(frame, feats, face_lms, pose_lms, indices, fps, detect_enabled, events):
    dbg = frame.copy()

    # HUD
    hud1 = f"EAR:{feats.get('ear',0):.3f}  MAR:{feats.get('mar',0):.3f}"
    hud2 = f"Fatigue:{indices['fatigue']:.0f}  Stress:{indices['stress']:.0f}"
    h, w = dbg.shape[:2]
    res  = f"{w}x{h}"
    cv2.putText(dbg, hud1, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(dbg, hud2, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2, cv2.LINE_AA)
    cv2.putText(dbg, f"FPS:{fps:.1f}  RES:{res}", (16, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2, cv2.LINE_AA)

    # FaceMesh 윤곽
    if face_lms:
        _polyline(dbg, face_lms, LEFT_EYE_RING,  (0,255,0),  closed=True, thickness=1)
        _polyline(dbg, face_lms, RIGHT_EYE_RING, (0,255,0),  closed=True, thickness=1)
        _polyline(dbg, face_lms, MOUTH_OUTER,    (255,200,0), closed=True, thickness=1)
        _polyline(dbg, face_lms, MOUTH_INNER,    (255,200,0), closed=True, thickness=1)
        for i in [1, 4, 33, 133, 263, 362, 13, 14]:
            x, y = _pt(dbg, face_lms, i)
            cv2.circle(dbg, (x, y), 2, (0,255,255), -1, cv2.LINE_AA)

    # 포즈(어깨 라인)
    if pose_lms:
        try:
            ls = pose_lms[11]; rs = pose_lms[12]
            p1 = (int(ls[0]*w), int(ls[1]*h))
            p2 = (int(rs[0]*w), int(rs[1]*h))
            cv2.line(dbg, p1, p2, (200, 100, 255), 2, cv2.LINE_AA)
            cv2.circle(dbg, p1, 3, (200, 100, 255), -1, cv2.LINE_AA)
            cv2.circle(dbg, p2, 3, (200, 100, 255), -1, cv2.LINE_AA)
        except Exception:
            pass

    # 이벤트 배지
    if events.get("blink"):
        cv2.putText(dbg, "BLINK", (w-140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    if events.get("yawn"):
        cv2.putText(dbg, "YAWN",  (w-135, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2, cv2.LINE_AA)

    # 감지 OFF 워터마크
    if not detect_enabled:
        overlay = dbg.copy()
        cv2.rectangle(overlay, (0, int(h*0.82)), (w, h), (50,50,50), -1)
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, dbg, 1-alpha, 0, dbg)
        cv2.putText(dbg, "DETECTION PAUSED", (16, h-24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 200, 255), 2, cv2.LINE_AA)

    return dbg

# 전역: WS 처리 잠깐 멈추는 스위치 (리포트 생성 시 사용)
PIPELINE_PAUSE = asyncio.Event()
PIPELINE_PAUSE.clear()

@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()

    # 설정 로드
    cam_config = CONFIG.get("camera", {})
    vision_config = CONFIG.get("vision", {})
    perf_config = CONFIG.get("performance", {})
    
    # 카메라 설정
    cam_width = cam_config.get("width", 640)
    cam_height = cam_config.get("height", 480)
    cam_fps = cam_config.get("fps", 20)
    max_num_faces = cam_config.get("max_num_faces", 3)
    
    # 비전 기능
    use_pnp = vision_config.get("use_pnp_headpose", True)
    use_target_tracking = vision_config.get("use_target_tracking", True)
    use_brightness = vision_config.get("use_brightness_check", True)
    
    logging.info(f"🎥 Camera: {cam_width}x{cam_height} @ {cam_fps}fps")
    logging.info(f"🔧 PnP: {use_pnp}, Tracking: {use_target_tracking}, Brightness: {use_brightness}")

    # Windows에서 카메라 점유 이슈가 있을 수 있어, dshow 우선시 옵션 허용
    try:
        cam = Camera(0, cam_width, cam_height, cam_fps, use_dshow=True).open()
    except Exception:
        cam = Camera(0, cam_width, cam_height, cam_fps).open()

    # FaceMesh (vis_test 기능 적용)
    fm = FaceMeshWrapper(
        use_pose=True,
        max_num_faces=max_num_faces,
        use_target_tracking=use_target_tracking
    )
    
    ev = EventState(fps=cam_fps)
    agg = WindowAggregator(window_sec=60)
    cal = Calibrator(warmup_sec=10, fps=cam_fps)  # 🆕 30→10초

    detect_enabled = True
    last_preview_ms = 0
    tbuf = collections.deque(maxlen=30)  # FPS

    camera_fail_cnt = 0
    
    # FPS 모니터링
    frame_times = collections.deque(maxlen=30)
    
    # 🆕 누적 통계 추적
    cumulative_stats = {
        "blink_count": 0,
        "yawn_count": 0,
        "nodding_count": 0,
        "fatigue_history": [],
        "stress_history": [],
        "perclos_history": [],
        "timestamps": []
    }
    
    # 🆕 DB 저장 타이머
    last_log_time = time.time()

    try:
        while True:
            if PIPELINE_PAUSE.is_set():
                await asyncio.sleep(0.05)
                continue

            # 컨트롤 수신(논블로킹)
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                try:
                    obj = json.loads(msg)
                    if obj.get("cmd") == "detect":
                        detect_enabled = bool(obj.get("enable"))
                except Exception:
                    pass
            except asyncio.TimeoutError:
                pass

            # 카메라 읽기 예외안전 + 자동 재오픈
            frame_start = time.time()
            try:
                frame = cam.read()
                camera_fail_cnt = 0
            except Exception as e:
                camera_fail_cnt += 1
                if camera_fail_cnt <= 3:
                    await asyncio.sleep(0.1)
                    try:
                        cam.close()
                    except:
                        pass
                    try:
                        cam = Camera(0, 1280, 720, 30, use_dshow=True).open()
                    except:
                        pass
                    continue
                # 상태만 알리고 루프 유지
                payload = {"ts": datetime.now(timezone.utc).isoformat(), "camera_error": str(e)}
                try:
                    await ws.send_text(json.dumps(payload, ensure_ascii=False))
                except WebSocketDisconnect:
                    break
                await asyncio.sleep(0.5)
                continue

            lm = fm.process(frame)
            feats = compute_all(
                frame, 
                lm, 
                use_pnp=use_pnp, 
                use_brightness=use_brightness
            )
            
            # FPS 측정
            frame_times.append(time.time())
            if len(frame_times) >= 2:
                duration = frame_times[-1] - frame_times[0]
                measured_fps = len(frame_times) / duration if duration > 0 else cam_fps
            else:
                measured_fps = cam_fps
            
            # 품질 업데이트
            feats["quality"]["fps"] = measured_fps

            # 개인 임계 자동화
            cal.consume(feats)
            if cal.ready:
                ev.th_close = cal.th_close
                ev.th_open  = cal.th_open
                ev.th_yawn  = cal.th_yawn

            if detect_enabled:
                events = ev.update(feats)
                agg.update(feats, events)
                snap = agg.snapshot()
                fused = {
                    "perclos":           snap.get("perclos", feats.get("perclos", 0.0)),
                    "yawn_rate_min":     snap.get("yawn_rate_min", 0.0),
                    "nodding_rate_min":  0.0,
                    "posture_angle_norm":snap.get("posture_angle_norm", feats.get("posture_angle_norm", 0.0)),
                    "headpose_var":      snap.get("headpose_var", feats.get("headpose_var", 0.0)),
                    "gaze_on_pct":       snap.get("gaze_on_pct", feats.get("gaze_on_pct", 0.7)),
                    "near_work":         snap.get("near_work", feats.get("near_work", 0.0)),
                    "facial_tension":    feats.get("facial_tension", 0.5),
                    "blink_var":         feats.get("blink_var", 0.2),
                }
                indices = compute_from_features(fused)
                events_out = events
                
                # 🆕 누적 카운팅
                if events.get("blink"):
                    cumulative_stats["blink_count"] += 1
                if events.get("yawn"):
                    cumulative_stats["yawn_count"] += 1
                if events.get("nodding"):
                    cumulative_stats["nodding_count"] += 1
                
                # 🆕 DB 저장 (10초마다)
                now_ts = time.time()
                if now_ts - last_log_time >= LOG_INTERVAL:
                    log_data = {
                        "perclos": fused["perclos"],
                        "yawn_rate": fused["yawn_rate_min"],
                        "posture": fused["posture_angle_norm"],
                        "headpose": fused["headpose_var"],
                        "fatigue": indices["fatigue"],
                        "stress": indices["stress"],
                        "blink": events.get("blink", False),
                        "yawn": events.get("yawn", False),
                        "nodding": events.get("nodding", False)
                    }
                    # 비동기로 DB 저장 (서버 멈춤 방지)
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, repo.save, log_data)
                    last_log_time = now_ts
                
                # 🆕 히스토리 저장 (최근 100개만)
                cumulative_stats["fatigue_history"].append(indices["fatigue"])
                cumulative_stats["stress_history"].append(indices["stress"])
                cumulative_stats["perclos_history"].append(fused["perclos"])
                cumulative_stats["timestamps"].append(datetime.now(timezone.utc).isoformat())
                
                if len(cumulative_stats["fatigue_history"]) > 100:
                    cumulative_stats["fatigue_history"].pop(0)
                    cumulative_stats["stress_history"].pop(0)
                    cumulative_stats["perclos_history"].pop(0)
                    cumulative_stats["timestamps"].pop(0)
            else:
                fused = {
                    "perclos": 0.0, "yawn_rate_min": 0.0, "nodding_rate_min": 0.0,
                    "posture_angle_norm": feats.get("posture_angle_norm", 0.0),
                    "headpose_var": 0.0, "gaze_on_pct": feats.get("gaze_on_pct", 0.7),
                    "near_work": feats.get("near_work", 0.0),
                    "facial_tension": feats.get("facial_tension", 0.5),
                    "blink_var": feats.get("blink_var", 0.2),
                }
                indices = {"fatigue": 0.0, "stress": 0.0}
                events_out = {"blink":0, "yawn":0, "nodding":0}

            # FPS
            tbuf.append(time.time())
            if len(tbuf) >= 2:
                fps = (len(tbuf)-1) / (tbuf[-1] - tbuf[0] + 1e-9)
            else:
                fps = 0.0

            # 디버그 오버레이 & 프리뷰
            dbg = draw_debug_overlay(frame, feats, lm.get("face_landmarks"), lm.get("pose_landmarks"),
                                     indices, fps=fps, detect_enabled=detect_enabled, events=events_out)

            frame_b64 = None
            now_ms = int(time.time() * 1000)
            if now_ms - last_preview_ms >= 250:
                last_preview_ms = now_ms
                ok, buf = cv2.imencode(".jpg", dbg, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    frame_b64 = base64.b64encode(buf).decode("ascii")

            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "features": {
                    "perclos": fused["perclos"],
                    "yawn_rate_min": fused["yawn_rate_min"],
                    "posture_angle_norm": fused["posture_angle_norm"],
                    "headpose_var": fused["headpose_var"],
                    "gaze_on_pct": fused["gaze_on_pct"],
                    "distance_cm": feats.get("distance_cm", 50.0),
                    "near_work": fused["near_work"]
                },
                "indices": indices,
                "events": events_out,
                "quality": feats.get("quality", {"lighting":0.0,"fps":0.0,"occlusion":0.0}),
                "frame_b64": frame_b64,
                "detect_enabled": detect_enabled,
                "fps": fps,
                "cumulative": cumulative_stats,  # 🆕 누적 통계
                "debug": {  # 🆕 디버그 정보
                    "face_detected": lm.get("face_landmarks") is not None,
                    "calibration_ready": cal.ready,
                    "calibration_progress": cal.get_progress(),
                    "ear": feats.get("ear", 0),
                    "mar": feats.get("mar", 0),
                    "head_pose": feats.get("head_pose"),
                    "brightness": feats.get("brightness", 0),
                    "fhp_info": feats.get("fhp_info")
                }
            }

            # WS 전송 예외 방어
            try:
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
            except WebSocketDisconnect:
                break

            await asyncio.sleep(0.01)
    finally:
        try: fm.close()
        except: pass
        try: cam.close()
        except: pass

@app.post("/report")
async def report(request: Request):
    # WS 파이프라인 잠깐 멈춤 -> 리포트 생성 중 프레임 송출 중단
    PIPELINE_PAUSE.set()
    await asyncio.sleep(0.15)
    try:
        payload = {}
        ctype = request.headers.get("content-type", "")
        try:
            if ctype.startswith("application/json"):
                payload = await request.json()
            elif ctype.startswith(("application/x-www-form-urlencoded","multipart/form-data")):
                form = await request.form()
                payload = {k: (json.loads(v) if isinstance(v, str) and v.strip().startswith(("{","[")) else v)
                           for k, v in form.items()}
            else:
                body = (await request.body() or b"").decode("utf-8","ignore").strip()
                if body.startswith("{"):
                    payload = json.loads(body)
        except Exception:
            payload = {}

        stats = payload.get("stats") or {}
        docs  = payload.get("docs")  or []

        # 🆕 트렌드 분석 추가
        try:
            # 1. DB에서 최근 12시간 데이터 가져오기
            history = await asyncio.to_thread(repo.get_data_for_analysis, hours=12)
            
            # 2. 분석기 실행
            analyzer = GraphAnalyzer()
            trend_text = analyzer.analyze(history, trend_window_min=10)  # 최근 10분 트렌드
            
            # 3. stats에 결과 주입 (LLM이 볼 수 있게)
            stats['trend_summary'] = trend_text
            
            # 로그 출력
            logging.info("-------- 트렌드 분석 결과 --------")
            logging.info(trend_text)
            logging.info("-------------------------------")
            
        except Exception as e:
            logging.error(f"Trend Analysis Failed: {e}")
            stats['trend_summary'] = "(트렌드 데이터 수집 중...)"

        # LLM 실행 (로컬 우선 / 최초 1회만 HF)
        txt = build_coaching_text(stats, docs)
        source = "local" if txt.startswith("[LLM:local]") else "fallback"
        text   = txt.replace("[LLM:local]\n","").replace("[LLM:fallback]\n","")

        return {"ok": True, "source": source, "text": text}
    finally:
        PIPELINE_PAUSE.clear()

@app.post("/chat")
async def chat(request: Request):
    """멀티턴 대화 엔드포인트"""
    PIPELINE_PAUSE.set()
    await asyncio.sleep(0.15)
    try:
        payload = await request.json()
        
        stats = payload.get("stats") or {}
        docs = payload.get("docs") or []
        conversation_history = payload.get("conversation_history") or []
        user_message = payload.get("user_message") or ""
        
        # 🆕 트렌드 분석 추가
        try:
            # 1. DB에서 최근 12시간 데이터 가져오기
            history = await asyncio.to_thread(repo.get_data_for_analysis, hours=12)
            
            # 2. 분석기 실행
            analyzer = GraphAnalyzer()
            trend_text = analyzer.analyze(history, trend_window_min=10)  # 최근 10분 트렌드
            
            # 3. stats에 결과 주입 (LLM이 볼 수 있게)
            stats['trend_summary'] = trend_text
            
            # 로그 출력
            logging.info("-------- 트렌드 분석 결과 (Chat) --------")
            logging.info(trend_text)
            logging.info("----------------------------------------")
            
        except Exception as e:
            logging.error(f"Trend Analysis Failed: {e}")
            stats['trend_summary'] = "(트렌드 데이터 수집 중...)"
        
        # 대화 히스토리를 포함하여 LLM 호출
        txt = build_coaching_text(stats, docs, conversation_history, user_message)
        source = "local" if txt.startswith("[LLM:local]") else "fallback"
        text = txt.replace("[LLM:local]\n","").replace("[LLM:fallback]\n","")
        
        return {"ok": True, "source": source, "text": text}
    finally:
        PIPELINE_PAUSE.clear()
