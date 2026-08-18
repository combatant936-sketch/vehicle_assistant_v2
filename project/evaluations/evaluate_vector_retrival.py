import sys
import os
import time
import shutil
import pandas as pd
from sqlitesearch import TextSearchIndex
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import csv
from dotenv import load_dotenv
from pathlib import Path
from tqdm.auto import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from embedder import Embedder
load_dotenv()
REPO_ROOT = Path.cwd()  # since your guard already ensures this is the repo root
PERSIST_DIR = REPO_ROOT /"project"/os.getenv("CHROMA_DB_TEST_DIR")
CHROMA_TEST_COLLECTION=os.getenv("CHROMA_TEST_COLLECTION")
DATA_FILE_PATH = REPO_ROOT /"project"/ os.getenv("DATA_FILE_PATH")
DATA_TEST_FILE_PATH=REPO_ROOT /"project"/ os.getenv("DATA_TEST_FILE_PATH")


def fetch_documents():
    """Load each CSV row as a Document for vector indexing."""
    df = pd.read_csv(DATA_FILE_PATH)
    documents = []
    for _, row in df.iterrows():
        content = "\n".join(f"{k}: {v}" for k, v in row.items())
        documents.append(Document(
            page_content=content,
            metadata={"source": DATA_FILE_PATH.as_posix(), "issue_id": row["issue_id"]}
        ))
    print(f"Loaded {len(documents)} documents")
    return documents



def create_chunks(documents,chunk_size=500,chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_or_load_vectorstore(documents):
    """
    Chroma-based vector store.
    If persist_dir exists -> load it.
    If not -> remove any stale/partial dir, then ingest from DATA_PATH.
    """
    if os.path.isdir(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
    
    embeddings = Embedder(path="models/Xenova/all-MiniLM-L6-v2")
    if os.path.exists(PERSIST_DIR):
        print(f"Found existing vectorstore at {PERSIST_DIR}, loading it.")
        return Chroma(
            collection_name=CHROMA_TEST_COLLECTION,
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
        collection_name=CHROMA_TEST_COLLECTION,
        persist_directory=PERSIST_DIR,
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore



def search_vector(q,_vectorstore,num_results=5):
    vectorstore = _vectorstore
    query = q
    vector_results = [
        normalize_vector_result(d)
        for d in vectorstore.similarity_search(query, k=num_results)
    ]
    return vector_results


def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt += 1
    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank]:
                total_score += 1 / (rank + 1)
                break
    return total_score / len(relevance_total)

def evaluate(ground_truth, search_function):
    relevance_total = []
    for q in tqdm(ground_truth):
        keywords = q["keywords"]
        results = search_function(q)
        relevance = [
            any(kw.lower() in d["content"].lower() for kw in keywords)
            for d in results
        ]
        relevance_total.append(relevance)
    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total)
    }
def normalize_vector_result(doc):
    return {
        "content": doc.page_content,
        "source": doc.metadata.get("source", ""),
        "type": doc.metadata.get("type", ""),
    }

if __name__ == "__main__":
    df_question = pd.read_json(DATA_TEST_FILE_PATH, lines=True)
    ground_truth = df_question.to_dict(orient="records")

    documents = fetch_documents()
    chunks = create_chunks(documents,500,200)

    _vectorstore=create_or_load_vectorstore(chunks)

    print("Ingestion complete")
    print(evaluate(ground_truth, lambda q: search_vector(q["question"],_vectorstore,20)))

