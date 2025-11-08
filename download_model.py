#!/usr/bin/env python3
import os
import sys
import shutil
import tempfile
from huggingface_hub import snapshot_download, hf_hub_download
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
else:
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
    except HFValidationError:
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

# --- вставлено: обеспечить наличие vocab.txt и config.json для f5-tts ---
from huggingface_hub import hf_hub_download
import json

# fallback-репо/подпапка, откуда брать vocab/config если их нет
FALLBACK_REPO = "SWivid/F5-TTS"
FALLBACK_SUB = "F5TTS_v1_Base"
token = HF_TOKEN  # берем из ранее определённой переменной

# vocab
vocab_path = os.path.join(LOCAL_MODEL_DIR, "vocab.txt")
if not os.path.exists(vocab_path):
    try:
        print("⚙️  vocab.txt not found — downloading fallback vocab...")
        fh = hf_hub_download(repo_id=FALLBACK_REPO, filename=f"{FALLBACK_SUB}/vocab.txt", repo_type="model", token=token)
        shutil.move(fh, vocab_path)
        print("✅ vocab.txt saved to", vocab_path)
    except Exception as e:
        print("❌ failed to fetch vocab.txt fallback:", e)

# config
config_path = os.path.join(LOCAL_MODEL_DIR, "config.json")
if not os.path.exists(config_path):
    try:
        print("⚙️  config.json not found — downloading fallback config...")
        fh = hf_hub_download(repo_id=FALLBACK_REPO, filename=f"{FALLBACK_SUB}/config.json", repo_type="model", token=token)
        shutil.move(fh, config_path)
        print("✅ config.json saved to", config_path)
    except Exception as e:
        print("❌ failed to fetch config.json fallback:", e)

# --- вывод подсказки для интерфейса ---
model_pt = os.path.join(LOCAL_MODEL_DIR, "model_last_inference.safetensors")
if not os.path.exists(model_pt):
    model_pt = os.path.join(LOCAL_MODEL_DIR, "model_last.pt")

default_config = {
    "dim": 1024,
    "depth": 22,
    "heads": 16,
    "ff_mult": 2,
    "text_dim": 512,
    "conv_layers": 4,
}
default_config_str = str(default_config).replace("'", '"')

if os.path.exists(model_pt):
    print("\n➡️ Use these values in interface:")
    print("Choose TTS Model: Custom")
    print(f"Model: {model_pt}")
    print(f"Vocab: {vocab_path if os.path.exists(vocab_path) else '(none, not required)'}")
    print(f"Config: {config_path if os.path.exists(config_path) else default_config_str}")
else:
    print(f"⚠️ Model checkpoint not found in folder — check contents of {LOCAL_MODEL_DIR}")

