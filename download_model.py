#!/usr/bin/env python3
import os
import sys
import shutil
import tempfile
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HFValidationError

# Переменные из окружения
HF_REPO = os.environ.get("HF_REPO", "Misha24-10/F5-TTS_RUSSIAN")
HF_SUBDIR = os.environ.get("HF_SUBDIR", "F5TTS_v1_Base_v2")
LOCAL_MODELS_DIR = os.environ.get("LOCAL_MODELS_DIR", "/models")
LOCAL_MODEL_DIR = os.path.join(LOCAL_MODELS_DIR, HF_SUBDIR)
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN") or None

# Убедимся, что базовая папка существует
os.makedirs(LOCAL_MODELS_DIR, exist_ok=True)

# Проверяем, скачана ли модель
if os.path.exists(LOCAL_MODEL_DIR) and os.listdir(LOCAL_MODEL_DIR):
    print(f"✅ Model already exists at {LOCAL_MODEL_DIR}")
    sys.exit(0)

print(f"⬇️ Downloading model '{HF_REPO}' (subfolder '{HF_SUBDIR}') to {LOCAL_MODEL_DIR}")

try:
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=LOCAL_MODEL_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=["*"],
        repo_type="model",
        subfolder=HF_SUBDIR,
        token=HF_TOKEN,
    )
    print(f"✅ Model downloaded successfully to {LOCAL_MODEL_DIR}")
except TypeError:
    # fallback для старых версий huggingface_hub без subfolder
    tmpdir = tempfile.mkdtemp(prefix="hf_repo_")
    try:
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=tmpdir,
            local_dir_use_symlinks=False,
            allow_patterns=["*"],
            repo_type="model",
            token=HF_TOKEN,
        )
        src = os.path.join(tmpdir, HF_SUBDIR)
        if not os.path.exists(src):
            # пытаемся найти папку внутри tmpdir
            candidates = [os.path.join(root, d)
                          for root, dirs, _ in os.walk(tmpdir)
                          for d in dirs if d == HF_SUBDIR]
            if candidates:
                src = candidates[0]
            else:
                raise FileNotFoundError(f"Subfolder '{HF_SUBDIR}' not found inside downloaded repo at {tmpdir}")
        shutil.move(src, LOCAL_MODEL_DIR)
        print(f"✅ Model moved to {LOCAL_MODEL_DIR}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
except HFValidationError:
    # fallback для HFValidationError
    tmpdir = tempfile.mkdtemp(prefix="hf_repo_")
    try:
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=tmpdir,
            local_dir_use_symlinks=False,
            allow_patterns=["*"],
            repo_type="model",
            token=HF_TOKEN,
        )
        src = os.path.join(tmpdir, HF_SUBDIR)
        if not os.path.exists(src):
            candidates = [os.path.join(root, d)
                          for root, dirs, _ in os.walk(tmpdir)
                          for d in dirs if d == HF_SUBDIR]
            if candidates:
                src = candidates[0]
            else:
                raise FileNotFoundError(f"Subfolder '{HF_SUBDIR}' not found inside downloaded repo at {tmpdir}")
        shutil.move(src, LOCAL_MODEL_DIR)
        print(f"✅ Model moved to {LOCAL_MODEL_DIR}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
except Exception as e:
    print("❌ Unexpected error during download:", e)
    raise
