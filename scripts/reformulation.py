import os
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL = "gemini-2.5-flash-preview-05-20"
CHECKPOINT_INTERVAL = 100

gclient = genai.Client(api_key=GOOGLE_API_KEY)

# reformulation rules and constraints 
RULES = {
    "r1": "Fix grammatical and spelling errors.",
    "r2": "Replace uncommon or advanced words with simpler synonyms, preserving original meaning and not altering proper nouns or titles.",
    "r3": "Append 'for kids' to the end of the query."
}
CONSTRAINTS = "Keep it under 21 words. Do not add new subject matter, opinions, or links."


def reformulate_single(query: str, rule_key: str) -> str:
    """
    Apply a single reformulation rule using Gemini, with debug prints.
    """
    print(f"[DEBUG] reformulate_single: rule={rule_key}, query='{query}'")
    system_instruction = (
        "You are, a query-rewriting assistant for children ages 6-13."
        " Below are the rules and constraints:\n"
        f"{rule_key}: {RULES[rule_key]}\n"
        f"Constraints: {CONSTRAINTS}\n"
        "Output only the rewritten query with no extra text."
    )
    prompt = f"Apply this rule: {RULES[rule_key]}\nOriginal query: {query}\nRewritten query:"
    try:
        response = gclient.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0
            )
        )
        rewritten = response.text.strip()
        print(f"[DEBUG] reformulate_single: got='{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"[ERROR] reformulate_single failed for rule={rule_key}, query='{query}', error={e}")
        raise


def run_reformulation(input_csv: str, output_csv: str, n: int):
    # Read CSV without headers, first col = queries, second col = tags 
    df = pd.read_csv(input_csv, header=None)
    queries = df.iloc[:, 0].dropna().astype(str).tolist()[:n]
    total = len(queries)
    print(f"[INFO] Starting reformulation for {total} queries (limit n={n})")

    records: List[Dict[str, str]] = []
    for idx, original in enumerate(queries, start=1):
        print(f"[INFO] Processing query {idx}/{total}: '{original}'")
        rec: Dict[str, str] = {"original_query": original}

        # Sequential full pipeline: r1 -> r2 -> r3
        try:
            step1 = reformulate_single(original, "r1")
            step2 = reformulate_single(step1, "r2")
            full = reformulate_single(step2, "r3")
            rec.update({
                "step1_query": step1,
                "step2_query": step2,
                "full_query": full
            })
        except Exception:
            print(f"[WARN] Sequential pipeline aborted at query {idx}")
            records.append(rec)
            break

        # Ablation: each rule on the original
        for rule in ["r1", "r2", "r3"]:
            try:
                abl_key = f"{rule}_abl_query"
                rec[abl_key] = reformulate_single(original, rule)
            except Exception:
                print(f"[WARN] Ablation for {rule} failed at query {idx}")
                rec[f"{rule}_abl_query"] = None

        records.append(rec)

        # Checkpoint save every CHECKPOINT_INTERVAL queries
        if idx % CHECKPOINT_INTERVAL == 0:
            print(f"[INFO] Checkpoint at {idx} queries: saving interim results to '{output_csv}'")
            pd.DataFrame(records).to_csv(output_csv, index=False)


    # Save results
    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    print(f"[INFO] Reformulation complete. Results saved to '{output_csv}'")


if __name__ == "__main__":
    run_reformulation(
        input_csv="ChildrenQueries.csv",
        output_csv="reformulated_queries_debug.csv",
        n=301
    )
