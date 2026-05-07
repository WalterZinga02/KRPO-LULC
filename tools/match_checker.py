import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment


Triple = Tuple[str, str, str]


# =========================
# CONFIG
# =========================

FILE_A = "GPT4omini_sample_results.txt"
FILE_B = "LLaMa3_sample_results.txt"
SENTENCES_FILE = "sentences.txt"

OUTPUT_FILE = "triple_matching_analysis.xlsx"

SUBJECT_THRESHOLD = 0.65
RELATION_THRESHOLD = 1.0
OBJECT_THRESHOLD = 0.65


# =========================
# NORMALIZATION / SIMILARITY
# =========================

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%.-]", "", text)
    return text


def text_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


def relation_similarity(a: str, b: str) -> float:
    return 1.0 if normalize_text(a) == normalize_text(b) else 0.0


def triple_similarity(t1: Triple, t2: Triple):
    sim_subject = text_similarity(t1[0], t2[0])
    sim_relation = relation_similarity(t1[1], t2[1])
    sim_object = text_similarity(t1[2], t2[2])

    return sim_subject, sim_relation, sim_object


# =========================
# PARSING
# =========================

def parse_triples_line(line: str) -> List[Triple]:
    line = line.strip()

    if not line:
        return []

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        print(f"Warning: invalid JSON line skipped:\n{line}")
        return []

    triples = []

    for item in data:
        if isinstance(item, list) and len(item) == 3:
            triples.append((str(item[0]), str(item[1]), str(item[2])))

    return triples


def read_triples_file(path: str) -> List[List[Triple]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [parse_triples_line(line) for line in lines]


def read_sentences_file(path: str) -> List[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


# =========================
# MATCHING
# =========================

def get_hungarian_pairs(triples_a: List[Triple], triples_b: List[Triple]):
    if not triples_a or not triples_b:
        return []

    similarity_matrix = np.zeros((len(triples_a), len(triples_b)))
    component_scores = {}

    for i, triple_a in enumerate(triples_a):
        for j, triple_b in enumerate(triples_b):

            sim_subject, sim_relation, sim_object = triple_similarity(
                triple_a,
                triple_b
            )

            similarity = (sim_subject + sim_relation + sim_object) / 3

            similarity_matrix[i, j] = similarity
            component_scores[(i, j)] = (
                sim_subject,
                sim_relation,
                sim_object
            )

    cost_matrix = 1 - similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    pairs = []

    for i, j in zip(row_ind, col_ind):
        sim_subject, sim_relation, sim_object = component_scores[(i, j)]

        is_match = (
            sim_subject >= SUBJECT_THRESHOLD
            and sim_relation >= RELATION_THRESHOLD
            and sim_object >= OBJECT_THRESHOLD
        )

        pairs.append({
            "triple_a": triples_a[i],
            "status": "MATCH" if is_match else "NOT_MATCH",
            "triple_b": triples_b[j],
            "sim_subject": sim_subject,
            "sim_relation": sim_relation,
            "sim_object": sim_object,
        })

    return pairs


# =========================
# MAIN
# =========================

def main() -> None:
    data_a = read_triples_file(FILE_A)
    data_b = read_triples_file(FILE_B)
    sentences = read_sentences_file(SENTENCES_FILE)

    if len(data_a) != len(data_b):
        raise ValueError(
            f"The two triple files have different numbers of lines: "
            f"{len(data_a)} vs {len(data_b)}"
        )

    if len(sentences) != len(data_a):
        raise ValueError(
            f"The sentence file and triple files have different numbers of lines: "
            f"{len(sentences)} vs {len(data_a)}"
        )

    rows = []

    for sentence_id, (sentence, triples_a, triples_b) in enumerate(
        zip(sentences, data_a, data_b),
        start=1
    ):
        pairs = get_hungarian_pairs(triples_a, triples_b)

        if not pairs:
            continue

        first_row = True

        for pair in pairs:
            rows.append({
                "sentence_id": sentence_id if first_row else "",
                "sentence": sentence if first_row else "",
                "triple_txt1": str(pair["triple_a"]),
                "status": pair["status"],
                "triple_txt2": str(pair["triple_b"]),
                "subject_similarity": round(pair["sim_subject"], 4),
                "relation_similarity": round(pair["sim_relation"], 4),
                "object_similarity": round(pair["sim_object"], 4),
            })

            first_row = False

    df = pd.DataFrame(rows)
    df.to_excel(OUTPUT_FILE, index=False)

    print(f"Saved Excel file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()