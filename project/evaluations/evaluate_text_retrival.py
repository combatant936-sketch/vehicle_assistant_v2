import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
sys.path.append(str(Path(__file__).resolve().parents[2]))
from project.ingests.ingest_fresh_or_load_text_search_data import load_or_build_text_index
import random

text_index=load_or_build_text_index()
REPO_ROOT = Path.cwd()  # since your guard already ensures this is the repo root
DATA_TEST_FILE_PATH=REPO_ROOT /"project"/ os.getenv("DATA_TEST_FILE_PATH")
df_question = pd.read_json(DATA_TEST_FILE_PATH, lines=True)
ground_truth = df_question.to_dict(orient="records")


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


def sqllite_search(query, boost=None):
    if boost is None:
        boost = {}

    results = text_index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=5
    )
    return [normalize_text_result(doc) for doc in results]
    

df_validation = df_question[:50]
df_test = df_question[50:]

gt_val = df_validation.to_dict(orient="records")
gt_test = df_test.to_dict(orient="records")




def simple_optimize(param_ranges, objective_function, n_iterations=10):
    best_params = None
    best_score = float("-inf")

    for _ in range(n_iterations):
        current_params = {}
        for field, (low, high) in param_ranges.items():
            current_params[field] = random.uniform(low, high)

        current_score = objective_function(current_params)

        if current_score > best_score:
            best_score = current_score
            best_params = current_params

    return best_params


def objective(boost_params):
    def search_function(q):
        return sqllite_search(q["question"], boost=boost_params)

    results = evaluate(gt_val, search_function)
    return results["hit_rate"] + results["mrr"]

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
if __name__ == "__main__":
    param_ranges = {
    "issue_name": (0.0, 3.0),
    "obd_code": (0.0, 3.0),
    "system": (0.0, 3.0),
    "component": (0.0, 3.0),
    "severity": (0.0, 3.0),
    "symptoms": (0.0, 3.0),
    "likely_causes": (0.0, 3.0),
    "diagnostic_steps": (0.0, 3.0),
    "diy_or_mechanic": (0.0, 3.0),
}
    # print("Evaluating without boost parameters:", evaluate(ground_truth, lambda q: sqllite_search(q["question"])))
    best_params = simple_optimize(param_ranges, objective, n_iterations=10)
    print("Best boost parameters:", best_params)
    print("Evaluating with best parameters:", evaluate(gt_test, lambda q: sqllite_search(q["question"],boost=best_params)))
    

    # print(sqllite_search("What does OBD code P0171 mean?", boost=best_params))



