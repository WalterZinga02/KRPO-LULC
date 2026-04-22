import re
import ast
import unicodedata
from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Any
from scipy.optimize import linear_sum_assignment

Triplet = Tuple[str, str, str]

# =========================
# 1. NORMALIZATION
# =========================

STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "by",
    "and", "or", "with", "during", "between", "from"
}


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_text(text: str) -> str:
    if text is None:
        return ""

    text = strip_accents(str(text).lower().strip())
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\s%]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_relation(rel: str) -> str:
    return normalize_text(rel)


def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    tokens = normalize_text(text).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# =========================
# 2. STRING SIMILARITY
# =========================

def containment_score(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b:
        return len(a) / len(b)
    if b in a:
        return len(b) / len(a)
    return 0.0


def sequence_similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def token_f1(a: str, b: str, remove_stopwords: bool = True) -> float:
    ta = tokenize(a, remove_stopwords=remove_stopwords)
    tb = tokenize(b, remove_stopwords=remove_stopwords)

    if not ta or not tb:
        return 0.0

    set_a = set(ta)
    set_b = set(tb)

    overlap = len(set_a & set_b)
    if overlap == 0:
        return 0.0

    precision = overlap / len(set_a)
    recall = overlap / len(set_b)
    return 2 * precision * recall / (precision + recall)


def entity_similarity(a: str, b: str) -> float:
    return max(
        token_f1(a, b, remove_stopwords=True),
        containment_score(a, b),
        sequence_similarity(a, b)
    )


def relation_similarity(a: str, b: str) -> float:
    a = normalize_relation(a)
    b = normalize_relation(b)

    if a == b:
        return 1.0

    sim = sequence_similarity(a, b)
    return sim if sim >= 0.85 else 0.0


# =========================
# 3. TRIPLET SIMILARITY
# =========================

def triplet_similarity(
    pred: Triplet,
    gold: Triplet,
    w_s: float = 0.4,
    w_r: float = 0.2,
    w_o: float = 0.4,
    cap_if_relation_wrong: bool = True
) -> Dict[str, float]:
    ps, pr, po = pred
    gs, gr, go = gold

    s_sim = entity_similarity(ps, gs)
    r_sim = relation_similarity(pr, gr)
    o_sim = entity_similarity(po, go)

    total = w_s * s_sim + w_r * r_sim + w_o * o_sim

    if cap_if_relation_wrong and r_sim == 0.0:
        total = min(total, 0.49)

    return {
        "subject": s_sim,
        "relation": r_sim,
        "object": o_sim,
        "total": total
    }


# =========================
# 4. MATCHING
# =========================

def build_score_matrix(
    preds: List[Triplet],
    golds: List[Triplet],
    w_s: float = 0.4,
    w_r: float = 0.2,
    w_o: float = 0.4
):
    matrix = []
    detailed = []

    for p in preds:
        row = []
        detail_row = []
        for g in golds:
            sims = triplet_similarity(p, g, w_s=w_s, w_r=w_r, w_o=w_o)
            row.append(sims["total"])
            detail_row.append(sims)
        matrix.append(row)
        detailed.append(detail_row)

    return matrix, detailed


def best_one_to_one_matching(
    preds: List[Triplet],
    golds: List[Triplet],
    w_s: float = 0.4,
    w_r: float = 0.2,
    w_o: float = 0.4
):
    if not preds or not golds:
        return []

    score_matrix, detail_matrix = build_score_matrix(preds, golds, w_s=w_s, w_r=w_r, w_o=w_o)
    cost_matrix = [[1.0 - s for s in row] for row in score_matrix]

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append({
            "pred_index": int(i),
            "gold_index": int(j),
            "pred": preds[i],
            "gold": golds[j],
            "subject_score": detail_matrix[i][j]["subject"],
            "relation_score": detail_matrix[i][j]["relation"],
            "object_score": detail_matrix[i][j]["object"],
            "total_score": detail_matrix[i][j]["total"]
        })

    return matches


# =========================
# 5. THRESHOLDED METRIC
# =========================

def evaluate_sentence(
    preds: List[Triplet],
    golds: List[Triplet],
    threshold: float = 0.65,
    w_s: float = 0.4,
    w_r: float = 0.2,
    w_o: float = 0.4
) -> Dict[str, Any]:
    matches = best_one_to_one_matching(preds, golds, w_s=w_s, w_r=w_r, w_o=w_o)
    accepted = [m for m in matches if m["total_score"] >= threshold]

    tp = len(accepted)
    fp = len(preds) - tp
    fn = len(golds) - tp

    precision = tp / len(preds) if preds else 0.0
    recall = tp / len(golds) if golds else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    avg_subject = sum(m["subject_score"] for m in accepted) / tp if tp else 0.0
    avg_relation = sum(m["relation_score"] for m in accepted) / tp if tp else 0.0
    avg_object = sum(m["object_score"] for m in accepted) / tp if tp else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "avg_subject_score": avg_subject,
        "avg_relation_score": avg_relation,
        "avg_object_score": avg_object,
        "matches": accepted
    }


# =========================
# 6. FILE READING
# =========================

def parse_triplets_line(line: str) -> List[Triplet]:
    line = line.strip()

    if not line:
        return []

    try:
        data = ast.literal_eval(line)
    except Exception as e:
        raise ValueError(f"Cannot parse line:\n{line}\nError: {e}")

    if not isinstance(data, list):
        raise ValueError(f"Line is not a list: {line}")

    triplets: List[Triplet] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"Invalid triplet item: {item}")
        s, r, o = item
        triplets.append((str(s), str(r), str(o)))

    return triplets


def load_triplets_file(filepath: str) -> List[List[Triplet]]:
    all_lines_triplets = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if line == "":
                all_lines_triplets.append([])
            else:
                try:
                    all_lines_triplets.append(parse_triplets_line(line))
                except Exception as e:
                    raise ValueError(f"Error in file {filepath}, line {line_num}: {e}")

    return all_lines_triplets


# =========================
# 7. DATASET EVALUATION
# =========================

def evaluate_dataset(
    preds_by_sentence: List[List[Triplet]],
    golds_by_sentence: List[List[Triplet]],
    threshold: float = 0.65,
    w_s: float = 0.4,
    w_r: float = 0.2,
    w_o: float = 0.4
) -> Dict[str, Any]:

    if len(preds_by_sentence) != len(golds_by_sentence):
        raise ValueError(
            f"Different number of lines: preds={len(preds_by_sentence)} vs gold={len(golds_by_sentence)}"
        )

    sentence_results = []

    total_tp = total_fp = total_fn = 0
    total_preds = 0
    total_golds = 0

    total_subject_sum = 0.0
    total_relation_sum = 0.0
    total_object_sum = 0.0
    total_match_count = 0

    for idx, (preds, golds) in enumerate(zip(preds_by_sentence, golds_by_sentence), start=1):
        res = evaluate_sentence(preds, golds, threshold=threshold, w_s=w_s, w_r=w_r, w_o=w_o)

        sentence_results.append({
            "sentence_id": idx,
            "pred_count": len(preds),
            "gold_count": len(golds),
            "f1": res["f1"]
        })

        total_tp += res["tp"]
        total_fp += res["fp"]
        total_fn += res["fn"]

        total_preds += len(preds)
        total_golds += len(golds)

        matches = res["matches"]
        total_match_count += len(matches)
        total_subject_sum += sum(m["subject_score"] for m in matches)
        total_relation_sum += sum(m["relation_score"] for m in matches)
        total_object_sum += sum(m["object_score"] for m in matches)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    avg_subject = total_subject_sum / total_match_count if total_match_count > 0 else 0.0
    avg_relation = total_relation_sum / total_match_count if total_match_count > 0 else 0.0
    avg_object = total_object_sum / total_match_count if total_match_count > 0 else 0.0

    return {
        "num_sentences": len(preds_by_sentence),
        "total_gold_triplets": total_golds,
        "total_pred_triplets": total_preds,
        "thresholded_micro": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "component_scores": {
            "subject": avg_subject,
            "relation": avg_relation,
            "object": avg_object
        },
        "sentence_results": sentence_results
    }


# =========================
# 8. PRETTY PRINT
# =========================

def print_results(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("THRESHOLDED FUZZY TRIPLET EVALUATION - GLOBAL RESULTS")
    print("=" * 60)

    print("\n[Dataset]")
    print(f"- Sentences:               {results['num_sentences']}")
    print(f"- Gold triplets:           {results['total_gold_triplets']}")
    print(f"- Predicted triplets:      {results['total_pred_triplets']}")

    thr = results["thresholded_micro"]
    print("\n[Thresholded Micro]")
    print(f"- Threshold:               {thr['threshold']:.2f}")
    print(f"- True Positives:          {thr['tp']}")
    print(f"- False Positives:         {thr['fp']}")
    print(f"- False Negatives:         {thr['fn']}")
    print(f"- Precision:               {thr['precision']:.4f}")
    print(f"- Recall:                  {thr['recall']:.4f}")
    print(f"- F1-score:                {thr['f1']:.4f}")

    comp = results["component_scores"]
    print("\n[Average Component Similarity on Accepted Matches]")
    print(f"- Subject similarity:      {comp['subject']:.4f}")
    print(f"- Relation similarity:     {comp['relation']:.4f}")
    print(f"- Object similarity:       {comp['object']:.4f}")


# =========================
# 9. MAIN
# =========================

if __name__ == "__main__":
    GOLD_FILE = "gold.txt"
    PREDS_FILE = "preds.txt"
    THRESHOLD = 0.65

    golds_by_sentence = load_triplets_file(GOLD_FILE)
    preds_by_sentence = load_triplets_file(PREDS_FILE)

    results = evaluate_dataset(
        preds_by_sentence=preds_by_sentence,
        golds_by_sentence=golds_by_sentence,
        threshold=THRESHOLD
    )

    print_results(results)