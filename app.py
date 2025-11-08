import gradio as gr
from transformers import pipeline

# Загружаем TTS-пайплайн с моделью F5-TTS_RUSSIAN
tts = pipeline("text-to-speech", model="Misha24-10/F5-TTS_RUSSIAN")

# Функция синтеза: принимает текст, возвращает кортеж (частота_дискретизации, массив_сэмплов)
def synthesize(text):
    output = tts(text)
    # Предполагаем, что output["audio"] – это numpy-массив, а output["sampling_rate"] – частота
    return (output["sampling_rate"], output["audio"])

# Создаём интерфейс с одним текстовым полем и аудио-выходом
iface = gr.Interface(
    fn=synthesize, 
    inputs="text", 
    outputs="audio", 
    title="F5-TTS Russian",
    description="Синтез русской речи с помощью модели F5-TTS_RUSSIAN"
)

# Запускаем сервер Gradio на 0.0.0.0:7860
iface.launch(server_name="0.0.0.0", server_port=7860)