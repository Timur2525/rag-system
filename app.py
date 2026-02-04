import gradio as gr
from src.rag_engine import query_rag
import os

def rag_interface(message, history):
    """Функция-генератор для Gradio (Streaming)."""

    if not message.strip():
        yield "Пожалуйста, введите вопрос."
        return

    if not os.path.exists("chroma_db"):
        yield "База знаний не найдена!"
        return

    try:
        response_generator, sources = query_rag(message)

        partial_message = ""

        for chunk in response_generator:
            partial_message += chunk
            yield partial_message

        if sources:
            partial_message += "\n\n**Источники:**\n" + "\n".join([f"- {s}" for s in sources])
            yield partial_message

    except Exception as e:
        yield f"Ошибка работы с сервером"

demo = gr.ChatInterface(
    fn=rag_interface,
    title="Math RAG",
    description="Ответ генерируется на основе загруженных pdf файлов",
    examples=[
        "Что такое предел последовательности?",
        "Сформулируй закон больших чисел.",
        "Как определяется производная?",
        "Что такое условная вероятность?"
    ],
)

if __name__ == "__main__":
    demo.launch()