# download_exaone_env.py
import os, sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError

def env(k, d=None):
    v = os.getenv(k, d)
    return v.strip() if isinstance(v, str) else v

def whoami(token):
    try:
        who = HfApi().whoami(token=token)
        print(f"[OK] Logged in as: {who.get('name') or who.get('email')}")
    except Exception:
        print("[WARN] whoami failed; if gated/private, you still need access approval.")

def try_download(repo_id, token):
    print(f"[TRY] download: {repo_id} -> HF cache (default)")
    path = snapshot_download(
        repo_id=repo_id,
        token=token,
        local_dir=None,                 # ← 기본 캐시 사용 (C:\Users\me\.cache\huggingface\hub)
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json","tokenizer.model","tokenizer_config.json",
            "model.safetensors","pytorch_model.bin","*.md",
        ],
    )
    print("[DONE] Cached at:", path)
    return path  # snapshots\<commit-hash> 경로

def main():
    if Path(".env").exists():
        load_dotenv(".env")
        print("[INFO] Loaded .env")

    token   = env("HUGGINGFACE_HUB_TOKEN")
    repo    = env("EXAONE_REPO_ID", "LGAI-EXAONE/EXAONE-4.0-1.2B-Instruct")
    fbk_repo= env("FALLBACK_REPO_ID", "kakaocorp/kanana-1.5-v-3b-instruct")

    if not token:
        print("[ERROR] HUGGINGFACE_HUB_TOKEN missing")
        sys.exit(1)

    whoami(token)

    try:
        p = try_download(repo, token)
        print("[INFO] Use this in server as EXAONE_LOCAL_PATH (repo id도 가능):", repo)
        return
    except (RepositoryNotFoundError, HfHubHTTPError) as e:
        print("[WARN] Main download failed:", e)

    try:
        p = try_download(fbk_repo, token)
        print("[INFO] Main unavailable. Use fallback for now:", fbk_repo)
    except Exception as e:
        print("[ERROR] Fallback download failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
