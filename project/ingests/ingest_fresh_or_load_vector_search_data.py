import sys
import os
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent.parent))
from embedder import Embedder
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]   # N = however many folders up to root
PERSIST_DIR = REPO_ROOT / "project" / os.getenv("CHROMA_DB_DIR")
DATA_FILE_PATH=REPO_ROOT / "project" / os.getenv("DATA_FILE_PATH")

CHROMA_COLLECTION=os.getenv("CHROMA_COLLECTION")


def fetch_documents():
    """Load each CSV row as a Document for vector indexing."""
    df = pd.read_csv(DATA_FILE_PATH)
    documents = []
    for _, row in df.iterrows():
        content = "\n".join(f"{k}: {v}" for k, v in row.items())
        documents.append(Document(
            page_content=content,
            metadata={"source": "data.csv", "issue_id": row["issue_id"]}
        ))
    print(f"Loaded {len(documents)} documents")
    return documents



def create_chunks(documents,chunk_size=500,chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks


def _build_vectorstore(documents):
    """
    Chroma-based vector store.
    If persist_dir exists -> load it.
    If not -> remove any stale/partial dir, then ingest from DATA_PATH.
    """
    
    embeddings = Embedder(path="models/Xenova/all-MiniLM-L6-v2")
    if os.path.exists(PERSIST_DIR):
        print(f"Found existing vectorstore at {PERSIST_DIR}, loading it.")
        return Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )

    # Not present -> clear any stale/partial leftover, then build fresh
    if os.path.isdir(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    print(f"No vectorstore found at {PERSIST_DIR}, ingesting data from {DATA_FILE_PATH}...")

    

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=PERSIST_DIR,
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore

def create_or_load_vectorstore():
    documents=fetch_documents()
    chunks=create_chunks(documents,500,200)
    vectorstore=_build_vectorstore(chunks)
    return vectorstore
