import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# --- Configuration ---
PDFS_PATH = "docs/"
FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "hkunlp/instructor-large"

def create_vector_store():
    """Loads PDFs, splits them, creates local embeddings, and saves to FAISS."""
    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Removing old index at {FAISS_INDEX_PATH}...")
        shutil.rmtree(FAISS_INDEX_PATH)

    print("Loading PDFs...")
    loader = PyPDFDirectoryLoader(PDFS_PATH)
    documents = loader.load()
    if not documents:
        print(f"No documents found in {PDFS_PATH}. Aborting.")
        return
    print(f"Loaded {len(documents)} document pages.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print(f"Initializing local embedding model '{EMBEDDING_MODEL}'...")
    print("This may take a few minutes to download the first time.")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Use CPU
        encode_kwargs={'normalize_embeddings': True} # Crucial for better similarity search
    )

    print("Creating FAISS index (this is CPU intensive)...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(FAISS_INDEX_PATH)
    print(f"\n✅ Vector store created successfully at: {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    create_vector_store()