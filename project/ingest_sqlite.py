from sqlitesearch import TextSearchIndex
import os
def load_index():
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
        db_path="project\issues.db"
    )
    return index