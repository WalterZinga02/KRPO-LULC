import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pandas as pd
from rapidfuzz import fuzz


Triple = Tuple[str, str, str]


# =========================
# CONFIG
# =========================

TRIPLES_FILE = "GPT4ominiFullCorpus.txt"
SENTENCES_FILE = "Lulc_dataset.txt"

OUTPUT_FILE = "recurring_triple_patterns.xlsx"

MODEL_NAME = "GPT4o-mini"

FINAL_SCORE_THRESHOLD = 0.70
REPRESENTATIVE_SCORE_THRESHOLD = 0.80

MIN_SUBJECT_SCORE = 0.80
MIN_OBJECT_SCORE = 0.80

MIN_PATTERN_FREQUENCY = 3
MIN_DISTINCT_SENTENCES = 2


# =========================
# NORMALIZATION
# =========================

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%.-]", "", text)
    return text


def relation_group_key(rel: str) -> str:
    r = normalize_text(rel).upper()

    if r in {"CAUSES", "AFFECTS"}:
        return "CAUSES_AFFECTS"

    return r


def relations_match(rel1: str, rel2: str) -> bool:
    return relation_group_key(rel1) == relation_group_key(rel2)


# =========================
# MATCHING SCORE
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


def match_score(triple1: Triple, triple2: Triple) -> Dict[str, Any]:
    src1, rel1, tgt1 = triple1
    src2, rel2, tgt2 = triple2

    if not relations_match(rel1, rel2):
        return {
            "is_match": False,
            "score": 0.0,
        }

    src_score = entity_score(src1, src2)
    tgt_score = entity_score(tgt1, tgt2)

    final_score = harmonic_score(src_score, tgt_score)

    is_match = (
        final_score >= FINAL_SCORE_THRESHOLD
        and src_score >= MIN_SUBJECT_SCORE
        and tgt_score >= MIN_OBJECT_SCORE
    )

    return {
        "is_match": is_match,
        "score": final_score,
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
# UNION-FIND
# =========================

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a


# =========================
# UTILS
# =========================

def triple_to_string(triple: Triple) -> str:
    return f"{triple[0]} --{triple[1]}-- {triple[2]}"


def get_cached_score(
    i: int,
    j: int,
    score_cache: Dict[Tuple[int, int], Dict[str, Any]]
) -> Dict[str, Any]:
    if i == j:
        return {
            "is_match": True,
            "score": 1.0,
        }

    key = (min(i, j), max(i, j))

    return score_cache.get(
        key,
        {
            "is_match": False,
            "score": 0.0,
        }
    )


def choose_representative_item(
    cluster_items: List[Dict[str, Any]],
    score_cache: Dict[Tuple[int, int], Dict[str, Any]]
) -> Dict[str, Any]:
    best_item = None
    best_avg_score = -1.0

    for item_i in cluster_items:
        scores = []

        for item_j in cluster_items:
            if item_i["global_id"] == item_j["global_id"]:
                continue

            result = get_cached_score(
                item_i["global_id"],
                item_j["global_id"],
                score_cache
            )

            scores.append(result["score"])

        avg_score = sum(scores) / len(scores) if scores else 1.0

        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_item = item_i

    return best_item


# =========================
# MAIN
# =========================

def main() -> None:
    triples_by_sentence = read_triples_file(TRIPLES_FILE)
    sentences = read_sentences_file(SENTENCES_FILE)

    if len(triples_by_sentence) != len(sentences):
        raise ValueError(
            f"Different number of lines: "
            f"{len(triples_by_sentence)} triple lines vs {len(sentences)} sentences"
        )

    all_items = []

    for sentence_id, (sentence, triples) in enumerate(
        zip(sentences, triples_by_sentence),
        start=1
    ):
        for triple_id, triple in enumerate(triples, start=1):
            all_items.append({
                "global_id": len(all_items),
                "sentence_id": sentence_id,
                "sentence": sentence,
                "triple_id": triple_id,
                "model": MODEL_NAME,
                "triple": triple,
                "relation_group": relation_group_key(triple[1]),
            })

    n = len(all_items)

    if n == 0:
        print("No triples found.")
        return

    print(f"Loaded triples: {n}")
    print("Computing pairwise similarities by relation group...")

    uf = UnionFind(n)
    score_cache = {}

    relation_groups = {}

    for item in all_items:
        relation_groups.setdefault(
            item["relation_group"],
            []
        ).append(item["global_id"])

    for relation_key, ids in sorted(
        relation_groups.items(),
        key=lambda x: len(x[1]),
        reverse=True
    ):
        print(f"Processing relation group {relation_key}: {len(ids)} triples")

        for pos_i in range(len(ids)):
            i = ids[pos_i]

            for pos_j in range(pos_i + 1, len(ids)):
                j = ids[pos_j]

                result = match_score(
                    all_items[i]["triple"],
                    all_items[j]["triple"]
                )

                score_cache[(min(i, j), max(i, j))] = result

                if result["is_match"]:
                    uf.union(i, j)

    clusters = {}

    for i, item in enumerate(all_items):
        root = uf.find(i)
        clusters.setdefault(root, []).append(item)

    raw_recurring_clusters = []

    for cluster in clusters.values():
        distinct_sentence_ids = {
            item["sentence_id"]
            for item in cluster
        }

        if (
            len(cluster) >= MIN_PATTERN_FREQUENCY
            and len(distinct_sentence_ids) >= MIN_DISTINCT_SENTENCES
        ):
            raw_recurring_clusters.append(cluster)

    raw_recurring_clusters = sorted(
        raw_recurring_clusters,
        key=lambda c: len(c),
        reverse=True
    )

    pattern_rows = []
    final_patterns_count = 0

    for cluster in raw_recurring_clusters:
        representative_item = choose_representative_item(cluster, score_cache)
        representative_triple = representative_item["triple"]
        representative_id = representative_item["global_id"]

        clean_cluster_items = []

        for item in cluster:
            similarity_to_representative = get_cached_score(
                representative_id,
                item["global_id"],
                score_cache
            )["score"]

            if similarity_to_representative >= REPRESENTATIVE_SCORE_THRESHOLD:
                clean_cluster_items.append((item, similarity_to_representative))

        distinct_sentence_count = len({
            item["sentence_id"]
            for item, _score in clean_cluster_items
        })

        if (
            len(clean_cluster_items) < MIN_PATTERN_FREQUENCY
            or distinct_sentence_count < MIN_DISTINCT_SENTENCES
        ):
            continue

        final_patterns_count += 1
        pattern_id = final_patterns_count

        clean_cluster_items = sorted(
            clean_cluster_items,
            key=lambda x: x[1],
            reverse=True
        )

        first_row = True

        for item, similarity_to_representative in clean_cluster_items:
            pattern_rows.append({
                "pattern_id": pattern_id if first_row else "",
                "representative_triple": triple_to_string(representative_triple) if first_row else "",
                "distinct_sentences": distinct_sentence_count if first_row else "",
                "triple": triple_to_string(item["triple"]),
                "sentence_id": item["sentence_id"],
                "sentence": item["sentence"],
                "similarity_to_representative": similarity_to_representative,
            })

            first_row = False

    df_patterns = pd.DataFrame(pattern_rows)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_patterns.to_excel(
            writer,
            sheet_name="Recurring_Patterns",
            index=False
        )

    print(f"\nSaved Excel file: {OUTPUT_FILE}")
    print(f"Edge threshold: {FINAL_SCORE_THRESHOLD}")
    print(f"Representative threshold: {REPRESENTATIVE_SCORE_THRESHOLD}")
    print(f"Minimum pattern frequency: {MIN_PATTERN_FREQUENCY}")
    print(f"Minimum distinct sentences: {MIN_DISTINCT_SENTENCES}")
    print(f"Total triples: {n}")
    print(f"Final recurring patterns found: {final_patterns_count}")


if __name__ == "__main__":
    main()