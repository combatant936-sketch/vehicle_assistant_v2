import sys
import os
import time
import shutil
import pandas as pd
from sqlitesearch import TextSearchIndex
from langchain_chroma import Chroma
from langchain_core.documents import Document
import csv
sys.path.append(os.path.abspath('..')) # Adds the parent directory to the path
from embedder import Embedder
PERSIST_DIR = os.getenv("CHROMA_DB_DIR")
SQLITESEARCHDB = os.getenv("SQLITESEARCHDB")
CHROMA_COLLECTION=os.getenv("CHROMA_COLLECTION")
DATA_PATH=os.getenv("DATA_PATH")
def load_or_build_text_index():
    """
    sqlitesearch-based index.
    If db_path exists -> load it.
    If not -> remove any stale/partial file at db_path, then ingest from DATA_PATH.
    """
    if os.path.exists(SQLITESEARCHDB):
        print(f"Found existing index at {SQLITESEARCHDB}, loading it.")
        return TextSearchIndex(
            text_fields=[
                "issue_name",
                "system",
                "component",
                "symptoms",
                "likely_causes",
                "diagnostic_steps",
            ],
            keyword_fields=["issue_id"],
            db_path=SQLITESEARCHDB,
        )

    # Not present -> make sure there's no partial/corrupt leftover, then build fresh
    if os.path.isdir(SQLITESEARCHDB):
        shutil.rmtree(SQLITESEARCHDB)
    elif os.path.isfile(SQLITESEARCHDB):
        os.remove(SQLITESEARCHDB)

    print(f"No index found at {SQLITESEARCHDB}, ingesting data from {DATA_PATH}...")

    index = TextSearchIndex(
        text_fields=[
            "issue_name",
            "system",
            "component",
            "symptoms",
            "likely_causes",
            "diagnostic_steps",
        ],
        keyword_fields=["issue_id"],
        db_path=SQLITESEARCHDB,
    )

    df = pd.read_csv(DATA_PATH)
    documents = df.to_dict(orient="records")

    for doc in documents:
        index.add(doc)
        print(f"""Added: {doc["issue_name"][:60]}...""")
        time.sleep(0.5)

    index.close()
    print(f"Done. Index saved to {SQLITESEARCHDB}")

    # reopen so caller gets a usable handle
    return TextSearchIndex(
        text_fields=[
            "issue_name",
            "system",
            "component",
            "symptoms",
            "likely_causes",
            "diagnostic_steps",
        ],
        keyword_fields=["issue_id"],
        db_path=SQLITESEARCHDB,
    )


def create_or_load_vectorstore():
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

    print(f"No vectorstore found at {PERSIST_DIR}, ingesting data from {DATA_PATH}...")

    documents = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            page_content = f"""
            Issue: {row['issue_name']} ({row['obd_code']})
            System: {row['system']} | Component: {row['component']} | Severity: {row['severity']}
            Symptoms: {row['symptoms']}
            Likely Causes: {row['likely_causes']}
            Diagnostic Steps: {row['diagnostic_steps']}
            Recommendation: {row['diy_or_mechanic']}
            """.strip()

            metadata = {
                "issue_id": row["issue_id"],
                "obd_code": row["obd_code"],
                "system": row["system"],
                "component": row["component"],
                "severity": row["severity"],
                "diy_or_mechanic": row["diy_or_mechanic"],
                "source": DATA_PATH,
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION,
        persist_directory=PERSIST_DIR,
    )

    return vectorstore