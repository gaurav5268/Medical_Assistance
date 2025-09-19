# rag_emb.py — using HuggingFace / sentence-transformers embeddings (no Gemini API key)
import os
import pickle
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document

# Use LangChain's HuggingFaceEmbeddings where available, otherwise fallback to sentence-transformers
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    HF_EMBED_CLASS = "langchain"
except Exception:
    HF_EMBED_CLASS = None

# Optional: direct sentence-transformers fallback
try:
    from sentence_transformers import SentenceTransformer
    S2T_AVAILABLE = True
except Exception:
    S2T_AVAILABLE = False

load_dotenv()
print("Environment variables loaded (Hugging Face mode)")

DATA_DIR = "data/"
OUT_DIR = "vectorstore/"

# Choose the HF model you want to use (local or hub). all-MiniLM-L6-v2 is fast & good.
HF_MODEL_NAME = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Create an embeddings wrapper compatible with LangChain / FAISS
if HF_EMBED_CLASS == "langchain":
    # If langchain has HuggingFaceEmbeddings available
    embeddings = HuggingFaceEmbeddings(model_name=HF_MODEL_NAME)
elif S2T_AVAILABLE:
    # Create a small wrapper object providing embed_documents for FAISS.from_documents
    s2_model = SentenceTransformer(HF_MODEL_NAME)

    class _LocalSentenceTransformerWrapper:
        def __init__(self, model):
            self.model = model

        def embed_documents(self, texts):
            # SentenceTransformers returns numpy arrays; convert lists
            embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return [e.tolist() for e in embs]

        # FAISS.from_documents might call embed_query in some flows
        def embed_query(self, text):
            v = self.model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
            return v.tolist()

    embeddings = _LocalSentenceTransformerWrapper(s2_model)
else:
    raise RuntimeError(
        "No embeddings implementation available. Install 'sentence-transformers' or upgrade langchain to provide HuggingFaceEmbeddings."
    )


def load_pdf(file_path):
    """Try extracting text using PyPDFLoader first, then UnstructuredPDFLoader."""
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if docs and len(docs[0].page_content.strip()) > 0:
            return docs
    except Exception as e:
        print(f"PyPDFLoader failed: {e}")

    try:
        loader = UnstructuredPDFLoader(file_path, mode="elements")
        docs = loader.load()
        if docs:
            # some loaders return empty page_content on some pages; filter them
            docs = [d for d in docs if d.page_content and d.page_content.strip()]
            if docs:
                return docs
    except Exception as e:
        print(f"UnstructuredPDFLoader failed: {e}")

    print(f"No extractable text found in: {file_path}")
    return []


recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
semantic_splitter = SentenceTransformersTokenTextSplitter(chunk_size=200, chunk_overlap=20)


def table_chunker(element):
    return [element]


def build_vectorstores():
    documents = []
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
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
            # semantic splitter returns list of Documents
            chunks = semantic_splitter.split_documents([doc])
        else:
            chunks = recursive_splitter.split_documents([doc])

        all_chunks.extend(chunks)

    print(f"Documents split into {len(all_chunks)} chunks")

    if not all_chunks:
        print("⚠️ No text chunks extracted. Skipping vectorstore creation.")
        return None, None

    # Build FAISS vectorstore (LangChain community wrapper)
    # FAISS.from_documents expects embeddings object with embed_documents / embed_query methods
    faiss_store = FAISS.from_documents(all_chunks, embeddings)
    faiss_store.save_local(os.path.join(OUT_DIR, "faiss"))
    print(f"FAISS vectorstore saved at {OUT_DIR}/faiss")

    # Build BM25 (classic)
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
