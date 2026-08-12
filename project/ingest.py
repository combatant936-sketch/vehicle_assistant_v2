import os
import pandas as pd
from minsearch import Index

DATA_PATH = os.getenv("DATA_PATH", "data/data.csv")

def load_index(data_path=DATA_PATH):
    df = pd.read_csv(data_path)
    documents = df.to_dict(orient="records")

    index = Index(
         text_fields=[
        "issue_name",
        "system",
        "component",
        "symptoms",
        "likely_causes",
        "diagnostic_steps",
    ],
    keyword_fields=["id"]
    )

    index.fit(documents)
    return index