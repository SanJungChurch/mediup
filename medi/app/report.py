from datetime import datetime
from rag.faiss_index import ensure_index, semantic_search
from llm.exaone import build_coaching_text

# 지표별 설명 및 기준값
METRIC_DESCRIPTIONS = {
    "perclos": {
        "name": "PERCLOS (눈 감김 비율)",
        "unit": "%",
        "normal": "< 15%",
        "warning": "15-30%",
        "danger": "> 30%",
        "description": "눈을 감고 있는 시간의 비율. 졸음 지표."
    },
    "yawn_rate_min": {
        "name": "하품 빈도",
        "unit": "회/분",
        "normal": "< 0.5",
        "warning": "0.5-1.0",
        "danger": "> 1.0",
        "description": "분당 하품 횟수. 피로/졸음 지표."
    },
    "posture_angle_norm": {
        "name": "자세 각도 (거북목)",
        "unit": "0-1",
        "normal": "< 0.3",
        "warning": "0.3-0.6",
        "danger": "> 0.6",
        "description": "머리가 앞으로 숙여진 정도. 거북목 위험도."
    },
    "headpose_var": {
        "name": "머리 움직임 변동",
        "unit": "분산",
        "normal": "< 0.1",
        "warning": "0.1-0.3",
        "danger": "> 0.3",
        "description": "머리 자세의 불안정성. 스트레스/집중력 저하 지표."
    },
    "gaze_on_pct": {
        "name": "시선 집중도",
        "unit": "%",
        "normal": "> 70%",
        "warning": "50-70%",
        "danger": "< 50%",
        "description": "화면을 보고 있는 시간 비율."
    },
    "near_work": {
        "name": "근거리 작업",
        "unit": "비율",
        "normal": "< 0.3",
        "warning": "0.3-0.6",
        "danger": "> 0.6",
        "description": "40cm 이내 근거리 작업 시간 비율. 눈 피로 위험."
    }
}

def get_metric_status(metric_name: str, value: float) -> tuple:
    """지표 상태 판정 - (상태, 이모지) 반환"""
    desc = METRIC_DESCRIPTIONS.get(metric_name)
    if not desc:
        return ("측정중", "⏳")
    
    # 간단한 휴리스틱 판정
    if metric_name == "perclos":
        if value < 0.15:
            return ("정상", "✅")
        elif value < 0.30:
            return ("주의", "⚠️")
        else:
            return ("위험", "🔴")
    
    elif metric_name == "yawn_rate_min":
        if value < 0.5:
            return ("정상", "✅")
        elif value < 1.0:
            return ("주의", "⚠️")
        else:
            return ("위험", "🔴")
    
    elif metric_name == "posture_angle_norm":
        if value < 0.3:
            return ("정상", "✅")
        elif value < 0.6:
            return ("주의", "⚠️")
        else:
            return ("위험", "🔴")
    
    elif metric_name == "gaze_on_pct":
        if value > 0.7:
            return ("정상", "✅")
        elif value > 0.5:
            return ("주의", "⚠️")
        else:
            return ("위험", "🔴")
    
    elif metric_name == "near_work":
        if value < 0.3:
            return ("정상", "✅")
        elif value < 0.6:
            return ("주의", "⚠️")
        else:
            return ("위험", "🔴")
    
    return ("측정중", "⏳")

def build_metrics_table(stats: Dict) -> str:
    """지표 테이블 생성 (Markdown)"""
    lines = ["## 📊 상세 지표 분석\n"]
    lines.append("| 지표 | 측정값 | 상태 | 설명 |")
    lines.append("|:-----|-------:|:----:|:-----|")
    
    for key in ["perclos", "yawn_rate_min", "posture_angle_norm", 
                "headpose_var", "gaze_on_pct", "near_work"]:
        if key not in stats:
            continue
        
        desc = METRIC_DESCRIPTIONS.get(key)
        if not desc:
            continue
        
        value = stats[key]
        status, emoji = get_metric_status(key, value)
        
        # 값 포맷팅
        if desc["unit"] == "%":
            value_str = f"{value * 100:.1f}%"
        elif desc["unit"] == "0-1":
            value_str = f"{value:.2f}"
        elif desc["unit"] == "회/분":
            value_str = f"{value:.2f}회"
        else:
            value_str = f"{value:.3f}"
        
        lines.append(
            f"| {desc['name']} | **{value_str}** | {emoji} {status} | {desc['description']} |"
        )
    
    return "\n".join(lines)

def build_report(session_id: str):
    # 1) 최근 지표 요약 (실제로는 MongoDB에서 가져옴)
    stats = {
        "avg_fatigue": 62.0,
        "avg_stress": 55.0,
        "perclos": 0.28,
        "yawn_rate_min": 0.7,
        "posture_angle_norm": 0.45,
        "headpose_var": 0.15,
        "gaze_on_pct": 0.68,
        "near_work": 0.3
    }
    
    # 2) RAG 검색
    ensure_index()
    docs = semantic_search("디지털 눈피로, 거북목, 시선, 시거리, 휴식 가이드", k=3)

    # 3) LLM 코칭 문장
    try:
        coaching = build_coaching_text(stats, docs)
        
        # fallback 체크
        if coaching.startswith("[LLM:fallback]"):
            coaching = coaching.replace("[LLM:fallback]\n", "")
            llm_status = "⚠️ 규칙 기반 (LLM 미사용)"
        else:
            coaching = coaching.replace("[LLM:local]\n", "")
            llm_status = "✅ AI 생성"
    except Exception as e:
        coaching = f"⚠️ 코칭 생성 실패: {e}"
        llm_status = "❌ 오류"

    # 4) 지표 테이블 생성
    metrics_table = build_metrics_table(stats)

    # 5) 최종 리포트 조합
    md = f"""# 💼 디지털 웰빙 리포트

**세션 ID:** `{session_id}`  
**작성 시각:** {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}  
**LLM 상태:** {llm_status}

---

## 🎯 AI 코칭 요약

{coaching}

---

{metrics_table}

---

## 📈 종합 지수

| 구분 | 점수 | 상태 |
|:-----|-----:|:----:|
| **피로도 (Fatigue)** | {stats['avg_fatigue']:.1f}/100 | {"🔴" if stats['avg_fatigue'] > 70 else "⚠️" if stats['avg_fatigue'] > 40 else "✅"} |
| **스트레스 (Stress)** | {stats['avg_stress']:.1f}/100 | {"🔴" if stats['avg_stress'] > 70 else "⚠️" if stats['avg_stress'] > 40 else "✅"} |

---

## 💡 즉시 실천 가이드

### 1️⃣ 눈 건강
- **20-20-20 규칙**: 20분마다 20초간 20피트(6m) 거리 보기
- **깜빡임 의식하기**: 의식적으로 완전한 깜빡임 실천

### 2️⃣ 자세 교정
- **모니터 높이**: 눈높이보다 10-15cm 아래
- **모니터 거리**: 팔 길이 (50-70cm) 유지
- **의자 세팅**: 발바닥이 바닥에 닿도록

### 3️⃣ 스트레칭
- **목 스트레칭**: 좌우 천천히 돌리기 (각 5회)
- **어깨 으쓱**: 어깨를 귀까지 올렸다 내리기 (3회)
- **손목 돌리기**: 양손 깍지 끼고 원 그리기 (10회)

---

## 📚 참고 자료

{chr(10).join([f"- [{d['title']}]({d.get('path', '#')})" for d in docs])}

---

<details>
<summary>📊 원본 데이터 (개발자용)</summary>

```json
{repr(stats)}
```

</details>
"""
    
    return {
        "summary_md": md, 
        "metrics": stats, 
        "evidence_doc_ids": [d["path"] for d in docs],
        "llm_status": llm_status
    }
