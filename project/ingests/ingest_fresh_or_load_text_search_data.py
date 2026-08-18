import sys
import os
import time
import shutil
import pandas as pd
from sqlitesearch import TextSearchIndex
from dotenv import load_dotenv
from pathlib import Path
from tqdm.auto import tqdm
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]   # N = however many folders up to root

DATA_FILE_PATH=REPO_ROOT / "project" / os.getenv("DATA_FILE_PATH")
SQLITESEARCHDB = REPO_ROOT / "project" / os.getenv("SQLITESEARCHDB")

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
                "obd_code",
                "system",
                "component",
                "severity",
                "symptoms",
                "likely_causes",
                "diagnostic_steps",
                "diy_or_mechanic",
            ],
            keyword_fields=["issue_id"],
            db_path=SQLITESEARCHDB,
        )

    # Not present -> make sure there's no partial/corrupt leftover, then build fresh
    if os.path.isdir(SQLITESEARCHDB):
        shutil.rmtree(SQLITESEARCHDB)
    elif os.path.isfile(SQLITESEARCHDB):
        os.remove(SQLITESEARCHDB)

    print(f"No index found at {SQLITESEARCHDB}, ingesting data from {DATA_FILE_PATH}...")

    index = TextSearchIndex(
        text_fields=[
            "issue_name",
            "obd_code",
            "system",
            "component",
            "severity",
            "symptoms",
            "likely_causes",
            "diagnostic_steps",
            "diy_or_mechanic",
        ],
        keyword_fields=["issue_id"],
        db_path=SQLITESEARCHDB,
    )

    df = pd.read_csv(DATA_FILE_PATH)
    documents = df.to_dict(orient="records")

    for doc in documents:
        index.add(doc)
        print(f"""Added: {doc["issue_name"][:60].encode('ascii', 'replace').decode()}...""")
        time.sleep(0.5)

    index.close()
    print(f"Done. Index saved to {SQLITESEARCHDB}")

    # reopen so caller gets a usable handle
    return TextSearchIndex(
        text_fields=[
            "issue_name",
            "obd_code",
            "system",
            "component",
            "severity",
            "symptoms",
            "likely_causes",
            "diagnostic_steps",
            "diy_or_mechanic",
        ],
        keyword_fields=["issue_id"],
        db_path=SQLITESEARCHDB,
    )