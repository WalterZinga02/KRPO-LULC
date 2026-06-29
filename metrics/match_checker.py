import json
import re
from itertools import combinations
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

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

SENTENCE_FILE_CANDIDATES = [
    INPUT_DIR / "dataset300.txt",
    INPUT_DIR / "lulc_sample.txt",
    INPUT_DIR / "valid_dataset.txt",
    INPUT_DIR / "lulc_dataset.txt",
]

OUTPUT_FILE = OUTPUT_DIR / "triple_matching_analysis.xlsx"

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


def find_sentence_file(expected_lines: int) -> Path:
    for path in SENTENCE_FILE_CANDIDATES:
        if path.exists() and len(read_sentences_file(path)) == expected_lines:
            return path

    raise FileNotFoundError(
        f"No sentence file with {expected_lines} lines found. Checked: "
        + ", ".join(str(path) for path in SENTENCE_FILE_CANDIDATES)
    )


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


def model_name_from_path(path: Path) -> str:
    name = path.stem
    suffix = "results"
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    return name


def make_sheet_name(model_a: str, model_b: str, used_names: set[str]) -> str:
    invalid_chars = '[]:*?/\\'
    sheet_name = f"{model_a}_vs_{model_b}"
    for char in invalid_chars:
        sheet_name = sheet_name.replace(char, "_")
    sheet_name = sheet_name[:31]

    base_name = sheet_name
    counter = 1
    while sheet_name in used_names:
        suffix = f"_{counter}"
        sheet_name = f"{base_name[:31 - len(suffix)]}{suffix}"
        counter += 1

    used_names.add(sheet_name)
    return sheet_name


def evaluate_pair(
    model_a: str,
    data_a: List[List[Triple]],
    model_b: str,
    data_b: List[List[Triple]],
    sentences: List[str],
):
    if len(data_a) != len(data_b):
        raise ValueError(
            f"{model_a} and {model_b} have different numbers of lines: "
            f"{len(data_a)} vs {len(data_b)}"
        )

    if len(sentences) != len(data_a):
        raise ValueError(
            f"The sentence file and triple files have different numbers of lines for "
            f"{model_a} vs {model_b}: sentences={len(sentences)} vs triples={len(data_a)}"
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

        sentence_matches = sum(
            1 for pair in pairs
            if pair["status"] == "MATCH"
        )

        sentence_union = len(triples_a) + len(triples_b) - sentence_matches

        if sentence_union > 0:
            sentence_jaccard = sentence_matches / sentence_union
        else:
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
                f"{model_a}_triple": str(pair["triple_a"]),
                "status": pair["status"],
                f"{model_b}_triple": str(pair["triple_b"]),
                "score": round(pair["score"], 4),
                "subject_similarity": round(pair["subject_similarity"], 4),
                "relation_similarity": round(pair["relation_similarity"], 4),
                "object_similarity": round(pair["object_similarity"], 4),
            })

            first_row = False

    overlap_a, overlap_b, _f1 = compute_metrics(
        total_a,
        total_b,
        total_matches
    )

    union = total_a + total_b - total_matches
    micro_jaccard = total_matches / union if union else 1.0

    mean_local_jaccard = (
        sum(local_jaccard_scores) / len(local_jaccard_scores)
        if local_jaccard_scores
        else 0.0
    )

    summary = {
        "model_a": model_a,
        "model_b": model_b,
        "threshold": FINAL_SCORE_THRESHOLD,
        "total_triples_a": total_a,
        "total_triples_b": total_b,
        "overlap_on_a": overlap_a,
        "overlap_on_b": overlap_b,
        "total_matches": total_matches,
        "micro_jaccard": micro_jaccard,
        "mean_local_jaccard": mean_local_jaccard,
        "total_pairs": total_pairs,
        "empty_empty_sentences": empty_empty_sentences,
    }

    return summary, pd.DataFrame(rows)


# =========================
# MAIN
# =========================

def main() -> None:
    result_files = sorted(INPUT_DIR.glob("*results.txt"))
    if len(result_files) < 2:
        raise FileNotFoundError(
            f"Expected at least two *results.txt files in {INPUT_DIR}"
        )

    data_by_model = {
        model_name_from_path(path): read_triples_file(path)
        for path in result_files
    }

    expected_lines = len(next(iter(data_by_model.values())))
    sentence_file = find_sentence_file(expected_lines)
    sentences = read_sentences_file(sentence_file)

    summary_rows = []
    pair_results = []

    for model_a, model_b in combinations(data_by_model.keys(), 2):
        summary, details_df = evaluate_pair(
            model_a=model_a,
            data_a=data_by_model[model_a],
            model_b=model_b,
            data_b=data_by_model[model_b],
            sentences=sentences,
        )
        summary_rows.append(summary)
        pair_results.append((model_a, model_b, details_df))

    OUTPUT_DIR.mkdir(exist_ok=True)

    used_sheet_names = {"Summary"}
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        for model_a, model_b, details_df in pair_results:
            sheet_name = make_sheet_name(model_a, model_b, used_sheet_names)
            details_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved Excel file: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
