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

FILE_A = "GPT4ominiresults.txt"
FILE_B = "LLaMa3results.txt"
SENTENCES_FILE = "dataset300.txt"

OUTPUT_FILE = "triple_matching_analysis.xlsx"

FINAL_SCORE_THRESHOLD = 0.55


# =========================
# NORMALIZATION
# =========================

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%.-]", "", text)
    return text


def relations_match(rel1: str, rel2: str) -> bool:
    r1 = normalize_text(rel1).upper()
    r2 = normalize_text(rel2).upper()

    if r1 == r2:
        return True

    equivalent_relations = {
        frozenset(["CAUSES", "AFFECTS"]),
    }

    return frozenset([r1, r2]) in equivalent_relations


# =========================
# PAIRING SIMILARITY
# Used only by Hungarian
# =========================

def basic_entity_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


def basic_relation_similarity(a: str, b: str) -> float:
    return 1.0 if relations_match(a, b) else 0.0


def pairing_similarity(triple_a: Triple, triple_b: Triple) -> float:
    sim_subject = basic_entity_similarity(triple_a[0], triple_b[0])
    sim_relation = basic_relation_similarity(triple_a[1], triple_b[1])
    sim_object = basic_entity_similarity(triple_a[2], triple_b[2])

    return (sim_subject + sim_relation + sim_object) / 3


# =========================
# FINAL MATCH SCORE
# =========================

def entity_score(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    r = fuzz.ratio(a, b) / 100.0
    ts = fuzz.token_sort_ratio(a, b) / 100.0
    pr = fuzz.partial_ratio(a, b) / 100.0

    return round(0.4 * ts + 0.4 * r + 0.2 * pr, 4)


def harmonic_score(src_score: float, tgt_score: float) -> float:
    if src_score + tgt_score == 0:
        return 0.0

    return round(
        2 * (src_score * tgt_score) / (src_score + tgt_score),
        4
    )


def match_score(triple1: Triple, triple2: Triple):
    src1, rel1, tgt1 = triple1
    src2, rel2, tgt2 = triple2

    relation_match = relations_match(rel1, rel2)

    if not relation_match:
        return {
            "is_match": False,
            "score": 0.0,
            "src_score": 0.0,
            "tgt_score": 0.0,
            "relation_match": False,
        }

    src_score = entity_score(src1, src2)
    tgt_score = entity_score(tgt1, tgt2)

    final_score = harmonic_score(src_score, tgt_score)

    is_match = final_score >= FINAL_SCORE_THRESHOLD

    return {
        "is_match": is_match,
        "score": final_score,
        "src_score": src_score,
        "tgt_score": tgt_score,
        "relation_match": True,
    }


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
    pair_scores = {}

    for i, triple_a in enumerate(triples_a):
        for j, triple_b in enumerate(triples_b):
            similarity_matrix[i, j] = pairing_similarity(triple_a, triple_b)
            pair_scores[(i, j)] = match_score(triple_a, triple_b)

    cost_matrix = 1 - similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    pairs = []

    for i, j in zip(row_ind, col_ind):
        result = pair_scores[(i, j)]

        pairs.append({
            "triple_a": triples_a[i],
            "status": "MATCH" if result["is_match"] else "NOT_MATCH",
            "triple_b": triples_b[j],
            "score": result["score"],
            "subject_similarity": result["src_score"],
            "relation_similarity": 1.0 if result["relation_match"] else 0.0,
            "object_similarity": result["tgt_score"],
        })

    return pairs


def compute_metrics(total_a: int, total_b: int, total_matches: int):
    precision = total_matches / total_a if total_a else 0.0
    recall = total_matches / total_b if total_b else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1


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

    total_a = 0
    total_b = 0
    total_matches = 0
    total_pairs = 0

    local_jaccard_scores = []
    empty_empty_sentences = 0

    for sentence_id, (sentence, triples_a, triples_b) in enumerate(
        zip(sentences, data_a, data_b),
        start=1
    ):
        total_a += len(triples_a)
        total_b += len(triples_b)

        pairs = get_hungarian_pairs(triples_a, triples_b)
        total_pairs += len(pairs)

        # =========================
        # LOCAL JACCARD PER SENTENCE
        # =========================

        sentence_matches = sum(
            1 for pair in pairs
            if pair["status"] == "MATCH"
        )

        sentence_union = len(triples_a) + len(triples_b) - sentence_matches

        if sentence_union > 0:
            sentence_jaccard = sentence_matches / sentence_union
        else:
            # Both models extracted zero triples:
            # agreement on a noisy/non-informative sentence
            sentence_jaccard = 1.0
            empty_empty_sentences += 1

        local_jaccard_scores.append(sentence_jaccard)

        if not pairs:
            continue

        first_row = True

        for pair in pairs:
            if pair["status"] == "MATCH":
                total_matches += 1

            rows.append({
                "sentence_id": sentence_id if first_row else "",
                "sentence": sentence if first_row else "",
                "triple_txt1": str(pair["triple_a"]),
                "status": pair["status"],
                "triple_txt2": str(pair["triple_b"]),
                "score": round(pair["score"], 4),
                "subject_similarity": round(pair["subject_similarity"], 4),
                "relation_similarity": round(pair["relation_similarity"], 4),
                "object_similarity": round(pair["object_similarity"], 4),
            })

            first_row = False

    df = pd.DataFrame(rows)
    df.to_excel(OUTPUT_FILE, index=False)

    precision, recall, f1 = compute_metrics(
        total_a,
        total_b,
        total_matches
    )

    mean_local_jaccard = (
        sum(local_jaccard_scores) / len(local_jaccard_scores)
        if local_jaccard_scores
        else 0.0
    )

    print(f"\nSaved Excel file: {OUTPUT_FILE}")

    print(f"Threshold:          {FINAL_SCORE_THRESHOLD}")

    print(f"Total triples A:    {total_a}")
    print(f"Total triples B:    {total_b}")

    print(f"Overlap on A:       {precision:.4f}")
    print(f"Overlap on B:       {recall:.4f}")

    print(f"Total matches:      {total_matches}")

    print(f"Mean local Jaccard: {mean_local_jaccard:.4f}")


if __name__ == "__main__":
    main()