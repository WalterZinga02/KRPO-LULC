import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment


Triple = Tuple[str, str, str]


# =========================
# CONFIG
# =========================

FILE_A = "GPT4omini_sample_results.txt"
FILE_B = "LLaMa3_sample_results.txt"

ENTITY_THRESHOLD = 0.35


# =========================
# NORMALIZATION
# =========================

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%.-]", "", text)
    return text


# =========================
# PAIRING SIMILARITY
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


def relations_match(rel1: str, rel2: str) -> bool:
    r1 = normalize_text(rel1).upper()
    r2 = normalize_text(rel2).upper()

    if r1 == r2:
        return True

    equivalent_relations = {
        frozenset(["CAUSES", "AFFECTS"]),
    }

    return frozenset([r1, r2]) in equivalent_relations


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


def match_score(
    triple1: Triple,
    triple2: Triple,
    entity_threshold: float = ENTITY_THRESHOLD,
):
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
    final_score = round((src_score + tgt_score) / 2, 4)

    is_match = (
        src_score >= entity_threshold
        and tgt_score >= entity_threshold
    )

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


# =========================
# MATCHING
# =========================

def count_matches(triples_a: List[Triple], triples_b: List[Triple]) -> int:

    if not triples_a or not triples_b:
        print("\n==============================")
        return 0

    similarity_matrix = np.zeros((len(triples_a), len(triples_b)))
    pair_scores = {}

    for i, triple_a in enumerate(triples_a):
        for j, triple_b in enumerate(triples_b):

            # Used only to decide Hungarian pairing
            similarity_matrix[i, j] = pairing_similarity(triple_a, triple_b)

            # Used only to decide MATCH / NOT_MATCH after pairing
            pair_scores[(i, j)] = match_score(triple_a, triple_b)

    cost_matrix = 1 - similarity_matrix

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = 0

    paired_a = set()
    paired_b = set()

    for i, j in zip(row_ind, col_ind):

        result = pair_scores[(i, j)]

        paired_a.add(i)
        paired_b.add(j)

        if result["is_match"]:
            matches += 1
            label = "MATCH"
        else:
            label = "NOT_MATCH"

        print(f"\n{label}")
        print(f"A: {triples_a[i]}")
        print(f"B: {triples_b[j]}")
        print(f"score:              {result['score']:.4f}")
        print(f"subject:            {result['src_score']:.4f}")
        print(f"object:             {result['tgt_score']:.4f}")
        print(f"relation:           {'1.0000' if result['relation_match'] else '0.0000'}")
        print(f"pairing_similarity: {similarity_matrix[i, j]:.4f}")

    print("\n--- UNPAIRED A ---")

    for i, triple in enumerate(triples_a):
        if i not in paired_a:
            print(f"A: {triple}")

    print("\n--- UNPAIRED B ---")

    for j, triple in enumerate(triples_b):
        if j not in paired_b:
            print(f"B: {triple}")

    print("\n==============================")

    return matches


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

    if len(data_a) != len(data_b):

        raise ValueError(
            f"The two files have different numbers of lines: "
            f"{len(data_a)} vs {len(data_b)}"
        )

    total_a = 0
    total_b = 0
    total_matches = 0

    for triples_a, triples_b in zip(data_a, data_b):

        total_a += len(triples_a)
        total_b += len(triples_b)

        total_matches += count_matches(
            triples_a,
            triples_b
        )

    precision, recall, f1 = compute_metrics(
        total_a,
        total_b,
        total_matches
    )

    print("\n=== MODEL GPT 4o mini vs MODEL LLaMa3 ===")

    print(f"model_a_file:       {FILE_A}")
    print(f"model_b_file:       {FILE_B}")
    print(f"entity_threshold:   {ENTITY_THRESHOLD}")
    print("relation_match:     exact")
    print("pairing_similarity: strict ratio-based")

    print(f"total_triples_a:    {total_a}")
    print(f"total_triples_b:    {total_b}")
    print(f"total_matches:      {total_matches}")

    print(f"precision:          {precision:.4f}")
    print(f"recall:             {recall:.4f}")
    print(f"f1:                 {f1:.4f}")


if __name__ == "__main__":
    main()