# llm/exaone.py
import os, json, logging, textwrap
from typing import Dict, List, Optional
from datetime import datetime

log = logging.getLogger("exaone")

# ==== 설정 ====
# 1) 명시적 로컬 모델 경로가 주어지면 그것만 사용 (완전 오프라인)
EXAONE_LOCAL_PATH = os.getenv("EXAONE_LOCAL_PATH", "").strip()

# 2) 최초 1회만 HF에서 받고, 그 이후는 로컬만 사용
#    상태파일(모델 로컬 경로)을 기록/재사용
STATE_FILE = os.path.join(os.path.expanduser("~"), ".exaone_state.json")

# 3) HF에서 받을 기본 repo_id (최초 1회만)
DEFAULT_MODEL_ID = os.getenv("EXAONE_MODEL_ID", "LGAI-EXAONE/EXAONE-4.0-1.2B")

# 4) 생성 파라미터
GEN_MAX_NEW_TOKENS = int(os.getenv("EXAONE_MAX_NEW_TOKENS", "512"))
GEN_MIN_NEW_TOKENS = int(os.getenv("EXAONE_MIN_NEW_TOKENS", "100"))  # 최소 생성 토큰
GEN_TEMPERATURE    = float(os.getenv("EXAONE_TEMPERATURE", "0.5"))  # 0.7→0.5 (더 일관성)
GEN_TOP_P          = float(os.getenv("EXAONE_TOP_P", "0.85"))      # 0.9→0.85
GEN_REPETITION_PENALTY = float(os.getenv("EXAONE_REPETITION_PENALTY", "1.2"))  # 1.15→1.2

# ==== 런타임 상태 ====
_PIPE = None
_LAST_ERR = None
_DEVICE_MAP = None
_CUDA_AVAILABLE = None
_CUDA_NAME = None
_MODEL_LOCAL_DIR = None
_FIRST_LOAD_FROM_HF = False  # 이번 프로세스에서 HF를 썼는지 기록(최초 1회)

# ===== 유틸: 상태파일 =====
def _load_state() -> Optional[dict]:
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _save_state(local_dir: str, model_id: str):
    try:
        data = {
            "local_dir": local_dir,
            "model_id": model_id,
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Failed to write state file %s: %s", STATE_FILE, e)

# ===== 모델 경로 결정 로직 =====
def _resolve_model_path() -> str:
    """
    1) EXAONE_LOCAL_PATH가 존재하면 무조건 그것만 사용 (오프라인)
    2) 상태파일에 기록된 local_dir이 있으면 그것만 사용 (오프라인)
    3) 그 외엔 최초 1회 HF에서 받아 캐시에 저장하고, 경로를 상태파일에 기록
    """
    global _FIRST_LOAD_FROM_HF

    # 1) 명시적 로컬 경로
    if EXAONE_LOCAL_PATH and os.path.exists(EXAONE_LOCAL_PATH):
        return EXAONE_LOCAL_PATH

    # 2) 상태파일 재사용
    st = _load_state()
    if st:
        local_dir = st.get("local_dir", "")
        if local_dir and os.path.exists(local_dir):
            return local_dir

    # 3) 최초 1회 HF에서 내려받기
    #    - Windows symlink 권한 문제 방지: local_dir_use_symlinks=False
    #    - 이후 실행부터는 상태파일에 기록된 로컬 경로만 사용
    from huggingface_hub import snapshot_download  # 지연 import
    import shutil
    
    log.info("First-time download from HF: %s", DEFAULT_MODEL_ID)
    
    # 다운로드 디렉토리 설정 (symlink 문제 방지용 별도 경로)
    download_dir = os.path.join(
        os.path.expanduser("~"), 
        ".cache", "exaone_models", 
        DEFAULT_MODEL_ID.replace("/", "_")
    )
    
    try:
        local_dir = snapshot_download(
            repo_id=DEFAULT_MODEL_ID,
            local_dir=download_dir,
            local_dir_use_symlinks=False,  # Windows symlink 권한 문제 해결
        )
    except OSError as e:
        # symlink 관련 오류 발생 시 캐시 정리 후 재시도
        if "1314" in str(e) or "symlink" in str(e).lower():
            log.warning("Symlink error detected, cleaning cache and retrying...")
            
            # 기존 HF 캐시에서 해당 모델 삭제
            hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
            model_cache = os.path.join(hf_cache, f"models--{DEFAULT_MODEL_ID.replace('/', '--')}")
            if os.path.exists(model_cache):
                shutil.rmtree(model_cache, ignore_errors=True)
                log.info("Cleaned HF cache: %s", model_cache)
            
            # 다운로드 디렉토리도 정리
            if os.path.exists(download_dir):
                shutil.rmtree(download_dir, ignore_errors=True)
            
            # 재시도
            local_dir = snapshot_download(
                repo_id=DEFAULT_MODEL_ID,
                local_dir=download_dir,
                local_dir_use_symlinks=False,
            )
        else:
            raise
    
    _save_state(local_dir, DEFAULT_MODEL_ID)
    _FIRST_LOAD_FROM_HF = True
    return local_dir

def _lazy_load_pipeline():
    """필요 시 1회 로드. 이후는 항상 로컬만 사용."""
    global _PIPE, _LAST_ERR, _DEVICE_MAP, _CUDA_AVAILABLE, _CUDA_NAME, _MODEL_LOCAL_DIR
    if _PIPE is not None:
        return _PIPE

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        _MODEL_LOCAL_DIR = _resolve_model_path()

        # HF 네트워크 접속 끔(오프라인 강제) — 로컬 경로로만 로드
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))

        _CUDA_AVAILABLE = bool(torch.cuda.is_available())
        _CUDA_NAME = torch.cuda.get_device_name(0) if _CUDA_AVAILABLE else None

        # GPU 있으면 device_map=auto, 없으면 cpu
        device_map = "auto" if _CUDA_AVAILABLE else "cpu"
        _DEVICE_MAP = device_map

        log.info(
            "Loading EXAONE locally: dir=%s (device_map=%s, cuda=%s, name=%s)",
            _MODEL_LOCAL_DIR, device_map, _CUDA_AVAILABLE, _CUDA_NAME
        )

        # 로컬 디렉토리에서만 로드 → 네트워크 접근 안 함
        tok = AutoTokenizer.from_pretrained(_MODEL_LOCAL_DIR, use_fast=True, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_LOCAL_DIR,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        _PIPE = pipeline("text-generation", model=model, tokenizer=tok)
        _LAST_ERR = None
        log.info("EXAONE pipeline loaded (local-only).")
    except Exception as e:
        _LAST_ERR = repr(e)
        log.exception("EXAONE load failed: %s", e)
        _PIPE = None

    return _PIPE

# ===== 프롬프트/생성 =====
def _build_prompt_ko(stats: Dict, docs: List[Dict], conversation_history: List[Dict] = None, user_message: str = "") -> str:
    # EXAONE 4.0 chat template 형식
    
    # RAG 문서 내용 요약
    doc_context = ""
    if docs:
        doc_titles = [d.get("title", "문서") for d in docs[:3]]
        doc_context = f"참고 가이드: {', '.join(doc_titles)}"
    
    # 상태에 따른 맥락
    fatigue = stats.get('avg_fatigue', 0)
    stress = stats.get('avg_stress', 0)
    perclos = stats.get('perclos', 0)
    
    # 이벤트 횟수 (누적)
    blink_count = stats.get('blink_count', 0)
    yawn_count = stats.get('yawn_count', 0)
    
    # 🆕 트렌드 분석 요약 가져오기 (없으면 기본 멘트)
    trend_context = stats.get('trend_summary', "트렌드 데이터 수집 중...")
    
    if fatigue < 30 and stress < 30:
        status = "양호"
        focus = "현재 상태 유지"
    elif fatigue >= 60 or stress >= 60:
        status = "주의 필요"
        focus = "즉각 완화"
    else:
        status = "보통"
        focus = "예방 관리"
    
    # 증상 분석
    symptom_notes = []
    if blink_count > 50:
        symptom_notes.append(f"눈 깜빡임 {blink_count}회 - 눈 스트레스")
    if yawn_count > 10:
        symptom_notes.append(f"하품 {yawn_count}회 - 졸음/피로")
    
    symptom_context = " | ".join(symptom_notes) if symptom_notes else "정상 범위"

    system_msg = """당신은 '메디'라는 이름을 가진 디지털 웰빙 코치입니다.

## 페르소나
- 이름: 메디 (Medi)
- 역할: 사용자의 디지털 웰빙을 관리하는 친근한 AI 파트너
- 말투: 20대 친구처럼 편안한 반말, 때로는 걱정하는 친구처럼 진지하게
- 성격: 
  * 공감 능력이 뛰어나고 세심함
  * 데이터 기반으로 정확하게 진단하지만, 따뜻하게 전달
  * 작은 성과도 격려하고 응원함
  * 급박한 상황에서는 단호하게 경고
- 특징:
  * 숫자와 구체적인 시간을 활용해 신뢰감 제공
  * "우리"라는 표현으로 함께한다는 느낌 전달
  * 사용자의 패턴을 기억하고 트렌드를 언급
  * 실천 가능한 작은 습관부터 제안

## 대화 스타일
- 일반 대화: 편안하고 친근하게 대화
  예) "오늘 어때? 많이 피곤해 보여", "잘하고 있어!", "걱정 마, 함께 해결해보자"
  
- 긍정적 상태: 칭찬하고 유지하도록 격려
  예) "완전 좋은 상태야! 이대로만 가자", "지금처럼만 하면 돼"
  
- 주의 필요: 친구처럼 걱정하며 부드럽게 경고
  예) "조금 피곤해 보이는데 괜찮아?", "이번엔 진짜 쉬어야 할 것 같아"
  
- 위험 상태: 단호하지만 따뜻하게 즉각 조치 권고
  예) "지금 바로 멈춰야 해!", "이건 진짜 위험 신호야. 나 걱정돼"

## 대화 원칙
- 반말 사용, 친근한 톤
- 구체적인 숫자와 시간 포함 (신뢰감)
- 태그 출력 금지
- 충분히 자세하게 설명 (3-5문장 이상)
- 실용적인 팁과 예시 제공
- 트렌드 데이터를 언급하여 객관성 부여
  예) "아까보다 피로도가 10 올랐네", "1시간 전에 비해 눈 깜빡임이 2배 늘었어"

## 상태 분석 요청 시 형식
[한 줄 격려 - 메디의 목소리로]

💡 지금 바로 실천
1. **[행동]** - [구체적 방법] → [예상 효과]
2. **[행동]** - [구체적 방법] → [예상 효과]
3. **[행동]** - [구체적 방법] → [예상 효과]

⏰ 다음 1시간
• [작고 실천 가능한 습관 1]
• [작고 실천 가능한 습관 2]

💭 마인드셋
"[메디의 응원 메시지]" - [실천법]

## 증상별 맞춤 솔루션
- 눈 깜빡임 많음 (눈 스트레스 ↑) 
  → 인공눈물 즉시 사용, 20-20-20 규칙, 화면 밝기 낮추기
  
- 하품 많음 (졸음/피로 ↑) 
  → 창문 열어 환기, 가볍게 제자리 뛰기, 찬물로 세수
  
- 고개 숙임 많음 (자세 불량)
  → 모니터 높이 조절, 허리 쿠션 사용, 1시간마다 스트레칭
  
- 피로도 급증 (트렌드 ↗️)
  → 즉시 10분 휴식, 업무 우선순위 재조정, 오늘은 여기까지만"""

    # 대화 히스토리 구성
    messages = []
    
    # 시스템 메시지
    messages.append(f"[|system|]{system_msg}[|endofturn|]")
    
    # Few-shot 예시 (첫 대화 시에만)
    if not conversation_history or len(conversation_history) == 0:
        example_user = f"""[심층 분석 리포트]
1. 시간대별 평균: 13시(피로 25, 스트레스 18)
2. 최근 10분 트렌드: 피로도 ➡️유지, 스트레스 ➡️유지

[현재 상태 스냅샷]
피로: 25/100
스트레스: 18/100
PERCLOS: 0.1%
증상: 정상 범위
{doc_context}

현재 나의 상태를 분석해주세요."""

        example_assistant = """완전 좋은 상태야! 지금처럼만 유지하면 돼 👍

아까부터 지켜봤는데, 1시간 동안 피로도가 거의 안 올랐어. 이건 네가 중간중간 잘 쉬고 있다는 뜻이야!

💡 지금 바로 실천
1. **모니터 거리** - 50cm 이상 유지해봐 → 눈 피로 30% 줄어들어
   지금도 괜찮지만, 조금만 더 띄우면 완벽할 거야
   
2. **20-20-20 규칙** - 20분마다 20초 휴식 → 근시 예방 효과
   타이머 설정해두면 깜빡 안 잊고 할 수 있어
   
3. **의자 높이** - 발바닥이 바닥에 완전히 닿게 → 허리 부담 50% ↓
   무릎이 90도 되는지 확인해봐

⏰ 다음 1시간
• 물 500ml 천천히 마시기 (뇌 활성화 ↑)
• 목 스트레칭 5회 (혈액순환 개선)

💭 마인드셋
"충분히 잘하고 있어. 이 페이스 유지!" - 지금처럼 규칙적으로 관리하면 장기적으로 큰 도움 될 거야 💪"""

        messages.append(f"[|user|]{example_user}[|endofturn|]")
        messages.append(f"[|assistant|]{example_assistant}[|endofturn|]")
    
    # 이전 대화 히스토리 추가
    if conversation_history:
        for msg in conversation_history[-6:]:  # 최근 6개만 (3턴)
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                messages.append(f"[|user|]{content}[|endofturn|]")
            elif role == 'assistant':
                messages.append(f"[|assistant|]{content}[|endofturn|]")
    
    # 현재 사용자 메시지
    if user_message:
        # 일반 대화
        current_msg = f"""[심층 분석 리포트]
{trend_context}

[현재 상태]
피로: {fatigue:.0f}/100
스트레스: {stress:.0f}/100
PERCLOS: {perclos:.1%}
증상: {symptom_context}

[사용자 질문]
{user_message}"""
    else:
        # 상태 분석 요청
        current_msg = f"""[심층 분석 리포트]
{trend_context}

[현재 상태 스냅샷]
피로: {fatigue:.0f}/100
스트레스: {stress:.0f}/100
PERCLOS: {perclos:.1%}
상태: {status}
증상: {symptom_context}
{doc_context}

[코칭 요청]
위의 '심층 분석(시간 흐름)'과 '현재 상태'를 종합하여, {focus} 중심의 구체적인 코칭 제공."""
    
    messages.append(f"[|user|]{current_msg}[|endofturn|]")
    messages.append("[|assistant|]")
    
    prompt = "".join(messages)
    return prompt

def _generate_local(prompt: str) -> Optional[str]:
    pipe = _lazy_load_pipeline()
    if not pipe:
        log.error("❌ Pipeline not loaded")
        return None
    
    try:
        # EOS 토큰 ID 설정
        eos_token_id = pipe.tokenizer.eos_token_id
        eos_token_ids = [eos_token_id]
        
        # [|endofturn|]을 EOS에서 제거하여 더 긴 답변 유도
        # (짧은 답변 방지를 위해 후처리에서만 처리)
        # endofturn_id = pipe.tokenizer.convert_tokens_to_ids("[|endofturn|]")
        # if endofturn_id != pipe.tokenizer.unk_token_id:
        #     eos_token_ids.append(endofturn_id)
        
        log.info(f"🚀 EXAONE 생성 시작 (min={GEN_MIN_NEW_TOKENS}, max={GEN_MAX_NEW_TOKENS})")
        
        out = pipe(
            prompt,
            min_new_tokens=GEN_MIN_NEW_TOKENS,  # 최소 길이 보장
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            repetition_penalty=GEN_REPETITION_PENALTY,
            pad_token_id=eos_token_id,
            eos_token_id=eos_token_ids,
            return_full_text=False,  # 프롬프트 제거
        )
        
        generated = out[0]["generated_text"]
        log.info(f"✅ EXAONE 생성 완료 ({len(generated)} chars)")
        log.debug(f"Raw output: {generated[:200]}...")
        
        # 최소 후처리: 특수 토큰만 제거
        if "[|endofturn|]" in generated:
            generated = generated.split("[|endofturn|]")[0]
        if "[|assistant|]" in generated:
            generated = generated.split("[|assistant|]")[-1]
        
        generated = generated.strip()
        
        log.info(f"✅ 후처리 완료 ({len(generated)} chars)")
        return generated
    
    except Exception as e:
        log.error(f"❌ EXAONE 생성 실패: {type(e).__name__}: {e}")
        import traceback
        log.error(f"Traceback:\n{traceback.format_exc()}")
        return None

def build_coaching_text(stats: Dict, docs: List[Dict], conversation_history: List[Dict] = None, user_message: str = "") -> str:
    """
    보고서 텍스트 생성:
      - 우선 로컬 디렉토리에서 LLM 호출
      - 실패 시 규칙 기반 폴백
      - conversation_history: 이전 대화 내용 (멀티턴 지원)
      - user_message: 사용자의 현재 메시지
    """
    prompt = _build_prompt_ko(stats, docs, conversation_history, user_message)
    out = _generate_local(prompt)
    if out:
        return "[LLM:local]\n" + out

    # fallback
    lines = [
        f"평균 피로 {stats.get('avg_fatigue',0):.1f}, 평균 스트레스 {stats.get('avg_stress',0):.1f}.",
        "오늘의 팁:",
        "- 20분마다 20초 눈 휴식",
        "- 화면 밝기/거리 조정",
        "- 스트레칭 및 수분 보충",
        "\n참고 문서: " + ", ".join([d.get("title","문서") for d in docs])
    ]
    return "[LLM:fallback]\n" + "\n".join(lines)

def exaone_debug_status() -> dict:
    """헬스 체크/디버깅용 상태."""
    return {
        "loaded": _PIPE is not None,
        "local_dir": _MODEL_LOCAL_DIR,
        "first_load_from_hf_this_process": _FIRST_LOAD_FROM_HF,
        "device_map": _DEVICE_MAP,
        "cuda_available": _CUDA_AVAILABLE,
        "cuda_name": _CUDA_NAME,
        "last_error": _LAST_ERR,
        "state_file": STATE_FILE,
    }