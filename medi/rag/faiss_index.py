import os, json
import numpy as np

INDEX_PATH = "rag_index.faiss"
META_PATH = "rag_index.meta.json"

def ensure_index():
    # 데모: 문서 몇 개를 메타에 넣어둠 (실전에서는 sentence-transformers로 임베딩 후 생성)
    if not os.path.exists(META_PATH):
        metas = [
            {"id":0,"title":"디지털 눈피로 가이드","path":"rag/docs/guide_des.md"},
            {"id":1,"title":"거북목 예방 체크리스트","path":"rag/docs/neck_checklist.md"},
            {"id":2,"title":"작업거리/조명 권장","path":"rag/docs/distance_lighting.md"},
        ]
        json.dump(metas, open(META_PATH,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def semantic_search(query: str, k: int = 3):
    # 데모: 단순 메타 반환 (실전: FAISS search 결과 반환)
    metas = json.load(open(META_PATH,"r",encoding="utf-8"))
    return metas[:k]
