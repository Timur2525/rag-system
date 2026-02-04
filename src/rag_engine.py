import re
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import CHROMA_PATH, EMBEDDING_MODEL, LM_STUDIO_URL, LM_STUDIO_API_KEY


def get_retriever():
    """Загружает векторную базу и возвращает ретривер."""
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=str(CHROMA_PATH), embedding_function=embedding_function)
    return db.as_retriever(search_kwargs={"k": 10})


def get_llm():
    """Подключается к локальному LM Studio."""
    return ChatOpenAI(
        base_url=LM_STUDIO_URL,
        api_key=LM_STUDIO_API_KEY,
        model="local-model",
        temperature=0.3,
    )

def remove_thinking_tags(text):
    """Удаляет блок <think>...</think> из текста."""
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return clean_text.strip()

def query_rag(question: str):
    """
    Основная функция:
    1. Ищет контекст в базе
    2. Формирует промпт
    3. Отправляет в LM Studio
    4. Возвращает поток (stream) ответа
    """
    retriever = get_retriever()
    llm = get_llm()

    docs = retriever.invoke(question)
    context_text = "\n\n---\n\n".join([d.page_content for d in docs])

    sources = list(set([d.metadata.get("source", "Unknown") for d in docs]))

    template = """Ты — помощник студента по математике. 
        Используй приведенный ниже контекст из учебников, чтобы ответить на вопрос.
        Если ответа нет в контексте, так и скажи. Не выдумывай факты.
        Отвечай развернуто, но по делу
        Контекст:
        {context}
        Вопрос: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"context": context_text, "question": question})
    clean_response = remove_thinking_tags(response)

    return clean_response, sources