import sys
import os
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[2]))
from project.ingests.ingest_fresh_or_load_text_search_data import load_or_build_text_index
from project.ingests.ingest_fresh_or_load_vector_search_data import create_or_load_vectorstore
text_index=load_or_build_text_index()
vector_store=create_or_load_vectorstore()
REPO_ROOT = Path.cwd()  # since your guard already ensures this is the repo root
DATA_TEST_FILE_PATH=REPO_ROOT /"project"/ os.getenv("DATA_TEST_FILE_PATH")

def normalize_text_result(doc):
    return {
        "content": (
            f"Issue: {doc.get('issue_name','')}\n"
            f"OBD Code: {doc.get('obd_code','')}\n"
            f"System: {doc.get('system','')}\n"
            f"Component: {doc.get('component','')}\n"
            f"Severity: {doc.get('severity','')}\n"
            f"Symptoms: {doc.get('symptoms','')}\n"
            f"Likely Causes: {doc.get('likely_causes','')}\n"
            f"Diagnostic Steps: {doc.get('diagnostic_steps','')}\n"
            f"DIY or Mechanic: {doc.get('diy_or_mechanic','')}"
        ),
        "source": "text_search"
    }
def normalize_vector_result(doc):
     return {
        "content": doc.page_content,
        "source": doc.metadata.get("source", ""),
        "type": doc.metadata.get("type", ""),
    }
def rrf(search_results,num_results, k=1):
        scores = {}
        doc_map = {}
        for results in search_results:
            for rank, doc in enumerate(results):
                key = doc["content"]
                if key not in scores:
                    scores[key] = 0
                    doc_map[key] = doc
                scores[key] += 1 / (k + rank + 1)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in ranked[:num_results]]

def text_search(query,num_results):
    boost = {'issue_name': 1.2008905478084222, 'obd_code': 2.098088922644685, 'system': 2.140262529797431, 'component': 2.3184782614891786, 'severity': 1.777390696163289, 'symptoms': 0.9667497485289172, 'likely_causes': 2.545800591641401, 'diagnostic_steps': 0.3219785830178361, 'diy_or_mechanic': 0.11644211604098953}


    results = text_index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=num_results
    )

    return results
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
def hybrid_search(query,num_results):
        all_text_results = []
        all_vector_results = []
        # Loop through all query variations and search
        text_results = [normalize_text_result(r) for r in text_search(query,num_results)]
        vector_results = [normalize_vector_result(d) for d in vector_store.similarity_search(query, k=num_results)]
        
        all_text_results.append(text_results)
        all_vector_results.append(vector_results)
        # RRF can take a list of lists of documents and fuse them perfectly
        fused_documents = rrf(all_text_results + all_vector_results,num_results=num_results)
        
        # Finally, rerank the best fused documents based on the original query
        
        return fused_documents


if __name__ == "__main__":
    df_question = pd.read_json(DATA_TEST_FILE_PATH, lines=True)
    ground_truth = df_question.to_dict(orient="records")

    print("Ingestion complete")
    print(evaluate(ground_truth, lambda q: hybrid_search(q["question"],20)))
