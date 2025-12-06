# rag_utils.py  (C:\Law_Chatbot 에 저장)
# knowledge_base 폴더의 .md/.txt 파일들을 로딩해서
# ChromaDB 벡터 저장소로 관리하고, 질의용 rag_search 함수를 제공.

import os
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader

BASE_DIR = r"C:\Law_Chatbot"
KB_DIR = os.path.join(BASE_DIR, "data_legal_rag")  # ← 여기만 변경
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")


def load_knowledge_documents() -> List:
    """
    data_legal_rag 폴더 아래의 모든 .txt/.md/.pdf 파일을 불러와
    LangChain Document 리스트로 반환.
    """
    docs = []
    for root, _, files in os.walk(KB_DIR):
        for fname in files:
            path = os.path.join(root, fname)

            if fname.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif fname.endswith(".txt") or fname.endswith(".md"):
                loader = TextLoader(path, encoding="utf-8")
            else:
                continue

            docs.extend(loader.load())

    return docs



def build_chroma_db(persist: bool = True):
    """
    knowledge_base의 문서들을 임베딩하여 ChromaDB 생성.
    persist=True 이면 디스크에 저장해두고, 아니면 메모리에서만 사용.
    """
    # 한국어/멀티태스크용 임베딩 모델 (필요에 따라 변경 가능)
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

    # 1) 원문 문서 로딩
    raw_docs = load_knowledge_documents()

    # 2) 슬라이싱 (chunk 단위로 잘라서 검색 품질 향상)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = splitter.split_documents(raw_docs)

    if persist:
        db = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
        )
        db.persist()
    else:
        db = Chroma.from_documents(
            docs,
            embedding=embeddings,
        )

    return db


def load_or_build_chroma_db():
    """
    이미 ChromaDB가 있으면 불러오고, 없으면 새로 구축.
    """
    if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        db = Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
        return db
    else:
        return build_chroma_db(persist=True)


def rag_search(query: str, db, top_k: int = 3) -> List[str]:
    """
    주어진 query와 가장 유사한 knowledge_base chunk들을 top_k개 반환.
    """
    results = db.similarity_search(query, k=top_k)
    return [r.page_content for r in results]
