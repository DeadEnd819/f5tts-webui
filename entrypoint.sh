#!/usr/bin/env bash
set -euo pipefail

# ... переменные среды и т.п. остаются здесь без изменений ...

# Вызываем скачивание модели
python3 /app/download_model.py

# Далее запускаем f5-tts
exec f5-tts_infer-gradio --host "${GRADIO_SERVER_NAME}" --port "${GRADIO_SERVER_PORT}"
