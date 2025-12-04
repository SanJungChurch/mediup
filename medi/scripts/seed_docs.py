# 간단 시드: RAG 문서 파일 생성
import os
os.makedirs("rag/docs", exist_ok=True)
open("rag/docs/guide_des.md","w",encoding="utf-8").write("디지털 눈피로 가이드\n- 20-20-20 규칙\n- 깜빡임 유도 팁\n")
open("rag/docs/neck_checklist.md","w",encoding="utf-8").write("거북목 예방 체크리스트\n- 모니터 높이/거리 조정\n- 스트레칭\n")
open("rag/docs/distance_lighting.md","w",encoding="utf-8").write("작업거리/조명 권장\n- 화면까지 40~70cm\n- 조도 균일\n")
print("seed docs created")
