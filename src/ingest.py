import os
import shutil
from markitdown import MarkItDown
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import DATA_DIR, CHROMA_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def load_and_convert_docs():
    """Сканирует папку data, конвертирует PDF в Markdown."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Создана папка {DATA_DIR}.")
        return []

    md = MarkItDown()
    documents = []

    print("Начинаю конвертацию документов...")
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DATA_DIR, filename)
            try:
                print(f"   Обработка: {filename} ...")
                result = md.convert(file_path)
                doc_obj = {
                    "content": result.text_content,
                    "source": filename
                }
                documents.append(doc_obj)
            except Exception as e:
                print(f"Ошибка с файлом {filename}: {e}")
    return documents


def split_documents(raw_docs):
    """Разбивает markdown-текст на чанки."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []
    for doc in raw_docs:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            all_chunks.append({
                "page_content": chunk,
                "metadata": {"source": doc["source"]}
            })

    return all_chunks


def create_vector_db():
    """Создает базу ChromaDB."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    raw_docs = load_and_convert_docs()
    if not raw_docs:
        print("Нет документов для обработки.")
        return
    print(f"✂Разбиение текста ({len(raw_docs)} файлов)...")
    chunks_data = split_documents(raw_docs)

    print(f"Генерация эмбеддингов и сохранение в ChromaDB ({len(chunks_data)} фрагментов)...")

    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    texts = [c["page_content"] for c in chunks_data]
    metadatas = [c["metadata"] for c in chunks_data]

    Chroma.from_texts(
        texts=texts,
        embedding=embedding_function,
        metadatas=metadatas,
        persist_directory=str(CHROMA_PATH)
    )
    print("База знаний успешно создана")

if __name__ == "__main__":
    create_vector_db()