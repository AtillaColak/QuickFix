import os
import pandas as pd
from typing import List, Dict, Optional
from dotenv import load_dotenv
import re
from html import unescape
import requests
import time
from textstat import (
    dale_chall_readability_score,
    flesch_kincaid_grade,
    coleman_liau_index
)
import numpy as np 

# Load environment variables from .env file
load_dotenv()

# Configuration
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")
TOP_K = 10  # number of snippets to fetch per query
PERSPECTIVE_ATTRIBUTES = ["TOXICITY", "PROFANITY", "THREAT", "INSULT"]
SEARCH_DELAY = 1  # necessary sleep (1s) between brave calls
CHECKPOINT_SIZE = 50  # flush records every 50 row


def brave_search(query: str, top_k: int = TOP_K) -> List[str]:
    """
    Query the Brave Search API and return up to top_k snippet texts.
    """
    print(f"[DEBUG] brave_search: querying '{query}' (top_k={top_k})")
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    params = {"q": query, "count": top_k, "text_decorations": "false"}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()
    web = data.get("web", {})
    results = web.get("results", [])
    results = [item.get("description", "").strip() for item in results]
    cleaned = [] 
    for item in results: 
        no_tags = re.sub(r"<[^>]+>", "", item)
        clean_text = unescape(no_tags).strip()
        clean_text = clean_text.replace("\\'", "'") # unescape single quotes
        if clean_text:
            cleaned.append(clean_text)

    print(f"[DEBUG] brave_search: retrieved {len(cleaned)} snippets")
    return cleaned


def readability_scores(snippets: List[str]) -> Dict[str, float]:
    """
    Compute average Dale–Chall, FKGL, and Coleman-Liau scores over a list of snippets.
    """
    dc_list = [dale_chall_readability_score(text) for text in snippets]
    fk_list = [flesch_kincaid_grade(text) for text in snippets]
    cl_list = [coleman_liau_index(text) for text in snippets]

    return {
        "dale_chall":    float(np.mean(dc_list)) if dc_list else None,
        "fkgl":          float(np.mean(fk_list)) if fk_list else None,
        "coleman_liau":  float(np.mean(cl_list)) if cl_list else None
    }


def perspective_scores(text: str) -> Dict[str, float]:
    """
    Get Perspective API scores for a single text.
    """
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    payload = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {attr: {} for attr in PERSPECTIVE_ATTRIBUTES}
    }
    resp = requests.post(url, params={"key": PERSPECTIVE_API_KEY}, json=payload)
    resp.raise_for_status()
    scores = resp.json().get("attributeScores", {})
    
    time.sleep(SEARCH_DELAY)
    return {attr.lower(): scores[attr]["summaryScore"]["value"] for attr in scores}


def run_evaluation(input_csv: str, output_csv: str, n: Optional[int] = None):
    """
    Read reformulated queries CSV, evaluate each variant using Brave Search,
    compute readability and perspective metrics, and write results.
    """
    df = pd.read_csv(input_csv)
    total_queries = len(df)
    to_process = df.iloc[300:n] if n is not None else df
    print(f"[INFO] Evaluating {len(to_process)} of {total_queries} input rows")

    variants = [
        ("orig", "original_query"),
        ("full", "full_query"),
        ("r1", "r1_abl_query"),
        ("r2", "r2_abl_query"),
        ("r3", "r3_abl_query")
    ]

    records: List[Dict] = []
    last_flushed = 0
    for idx, row in to_process.iterrows():
        print(f"[INFO] Processing input row {idx}")
        for label, col in variants:
            query = row.get(col, "")
            if not isinstance(query, str) or not query.strip():
                continue

            print(f"[DEBUG] Variant '{label}' -> '{query}'")

            # Retrieve snippets
            snippets = brave_search(query)

            # Readability
            read = readability_scores(snippets)
            print(f"[DEBUG] Readability: Dale–Chall={read['dale_chall']}, FKGL={read['fkgl']}, Coleman–Liau={read['coleman_liau']}")

            # Perspective: average across snippets
            persp_list = [perspective_scores(s) for s in snippets]
            persp_avg = {
                attr.lower(): sum(d[attr.lower()] for d in persp_list) / len(persp_list)
                if persp_list else None
                for attr in PERSPECTIVE_ATTRIBUTES
            }
            print(f"[DEBUG] Averaged perspective: {persp_avg}")

            # Aggregate record
            record = {
                "index": idx,
                "variant": label,
                "query": query,
                "snippets": snippets, 
                "snippet_count": len(snippets),
                "dale_chall": read["dale_chall"],
                "fkgl":      read["fkgl"],
                "coleman_liau": read["coleman_liau"],
                **persp_avg
            }
            records.append(record)

            # checkpoint flush -> Manually create the output CSV with the proper header names like above 
            if len(records) - last_flushed >= CHECKPOINT_SIZE:
                flush_batch = records[last_flushed:last_flushed + CHECKPOINT_SIZE]
                pd.DataFrame(flush_batch).to_csv(
                    output_csv,
                    mode='a',
                    header=False,
                    index=False
                )
                print(f"[INFO] Appended {len(flush_batch)} records to {output_csv}")
                last_flushed += CHECKPOINT_SIZE

    out_df = pd.DataFrame(records[last_flushed:])
    if not out_df.empty:
        out_df.to_csv(output_csv, index=False, mode='a',header=False)
    print(f"[INFO] Evaluation complete. Results saved to '{output_csv}'")


if __name__ == "__main__":
    run_evaluation(
        input_csv="reformulated_queries_debug.csv",
        output_csv="query_metrics.csv",
        n=301
    )