# RuleVision (규칙기반 피로·스트레스 스크리닝 MVP)

이 프로젝트는 웹캠/카메라 입력으로부터 눈(EAR/PERCLOS), 하품(MAR), 머리자세/거북목(FHP), 시선/시거리, 품질 메타를 추출하고
**규칙기반 산식**으로 피로/스트레스 지수를 산출하여 웹 대시보드에 표시합니다.
리포트 생성 시에는 **RAG(FAISS)** + **LLM(프롬프트만, EXAONE 연동 위치 표기)**를 이용합니다.

## 빠른 실행

```bash
# 1) 가상환경 활성화 후 의존성 설치 (이미 설치됨 가정)
# 2) 서버 실행
uvicorn app.server:app --reload --port 8000

# 3) 브라우저 열기
# http://localhost:8000
```

## 다음 단계
- `core/` 모듈의 TODO를 채우면 실시간 동작합니다.
- Windows에서 pip로 faiss 설치가 어려울 수 있으니, conda-forge로 faiss-cpu를 설치하거나 WSL2를 권장합니다.
