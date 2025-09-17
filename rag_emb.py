import os
import pickle
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document

load_dotenv()
API = os.getenv("GEMINI_API_KEY")

print("Environment variables loaded")

DATA_DIR = "data/"
OUT_DIR = "vectorstore/"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=API
)

def load_pdf(file_path):
    """Try extracting text using PyPDFLoader first, then UnstructuredPDFLoader."""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if docs and len(docs[0].page_content.strip()) > 0:
            return docs
    except:
        pass

    try:
        loader = UnstructuredPDFLoader(file_path, mode="elements")
        docs = loader.load()
        if docs:
            return docs
    except:
        pass

    print(f"No extractable text found in: {file_path}")
    return []


recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
semantic_splitter = SentenceTransformersTokenTextSplitter(chunk_size=200, chunk_overlap=20)

def table_chunker(element):
    return [element]


def build_vectorstores():
    documents = []
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    print(f"Found {len(files)} PDF files in {DATA_DIR}")

    for i, file_path in enumerate(files, 1):
        file_docs = load_pdf(file_path)
        documents.extend(file_docs)
        print(f"   Processed file {i}: {file_path} -> {len(file_docs)} pages")

    print(f"Total loaded documents: {len(documents)}")

    all_chunks = []
    for doc in documents:
        text = doc.page_content.strip()
        if "|" in text or "----" in text or "\t" in text:
            chunks = table_chunker(doc)
        elif len(text.split(".")) < 5:
            chunks = semantic_splitter.split_documents([doc])
        else:
            chunks = recursive_splitter.split_documents([doc])

        all_chunks.extend(chunks)

    print(f"Documents split into {len(all_chunks)} chunks")

    # Build FAISS
    faiss_store = FAISS.from_documents(all_chunks, embeddings)
    faiss_store.save_local(os.path.join(OUT_DIR, "faiss"))
    print(f"FAISS vectorstore saved at {OUT_DIR}/faiss")

    # Build BM25
    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = 5

    # Save BM25 retriever (pickled)
    bm25_path = os.path.join(OUT_DIR, "bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"BM25 retriever saved at {bm25_path}")

    print("\nBuild Summary:")
    print(f"Files processed : {len(files)}")
    print(f"Total chunks    : {len(all_chunks)}")
    print("Vectorstores created successfully!")

    return faiss_store, bm25_retriever


if __name__ == "__main__":
    build_vectorstores()
