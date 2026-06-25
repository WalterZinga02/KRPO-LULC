import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer


Triple = Tuple[str, str, str]


# =========================
# CONFIG
# =========================

TRIPLES_FILE = "LLaMa3FullCorpus.txt"
SENTENCES_FILE = "Lulc_dataset.txt"

MODEL_NAME = "GPT4o-mini"

SIMILARITY_MODE = "embedding"  # "fuzzy" or "embedding"

OUTPUT_FILE = f"recurring_triple_patterns_{SIMILARITY_MODE}.xlsx"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FUZZY_EDGE_THRESHOLD = 0.70
FUZZY_REPRESENTATIVE_THRESHOLD = 0.80

EMBEDDING_EDGE_THRESHOLD = 0.70
EMBEDDING_REPRESENTATIVE_THRESHOLD = 0.75

MIN_SUBJECT_SCORE = 0.75 #0.8 for fuzzy, 0.75 for embedding
MIN_OBJECT_SCORE = 0.75 #0.8 for fuzzy, 0.75 for embedding

MIN_PATTERN_FREQUENCY = 4
MIN_DISTINCT_SENTENCES = 2


# =========================
# NORMALIZATION
# =========================

def normalize_text(text: str) -> str:
    """
    Operations:
    - lowercase conversion
    - whitespace normalization
    - removal of punctuation symbols
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s%.-]", "", text)
    return text


def relation_group_key(rel: str) -> str:    # Grouping relations by their normalized form, with special handling for "CAUSES" and "AFFECTS"
    r = normalize_text(rel).upper()

    if r in {"CAUSES", "AFFECTS"}:
        return "CAUSES_AFFECTS"

    return r


def relations_match(rel1: str, rel2: str) -> bool:
    return relation_group_key(rel1) == relation_group_key(rel2)


# =========================
# FUZZY SCORE
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

    """
        Combines subject and object similarity using
        the harmonic mean.

        Used for both fuzzy and embedding similarity modes
    """

    if src_score + tgt_score == 0:
        return 0.0

    return round(
        2 * (src_score * tgt_score) / (src_score + tgt_score),
        4
    )


def fuzzy_pair_score(triple1: Triple, triple2: Triple) -> Dict[str, float]:
    src1, _rel1, tgt1 = triple1
    src2, _rel2, tgt2 = triple2

    src_score = entity_score(src1, src2)
    tgt_score = entity_score(tgt1, tgt2)
    final_score = harmonic_score(src_score, tgt_score)

    return {
        "score": final_score,
        "src_score": src_score,
        "tgt_score": tgt_score,
    }


# =========================
# EMBEDDING SCORE
# =========================

def build_embedding_text(text: str) -> str:
    return normalize_text(text)


def cosine_from_normalized(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2))


def embedding_pair_score(
    i: int,
    j: int,
    subject_embeddings: np.ndarray,
    object_embeddings: np.ndarray
) -> Dict[str, float]:
    """
    Semantic similarity module (M2).

    Embeddings are generated using the
    Sentence-Transformers model: all-MiniLM-L6-v2

    For each triple we compute:
    - cosine similarity between subject embeddings
    - cosine similarity between object embeddings

    The final triple similarity is obtained using
    the harmonic mean of the two cosine similarities.

    The clustering pipeline remains identical to the fuzzy baseline;
    only the similarity computation module changes.
    """
    src_score = cosine_from_normalized(
        subject_embeddings[i],
        subject_embeddings[j]
    )

    tgt_score = cosine_from_normalized(
        object_embeddings[i],
        object_embeddings[j]
    )

    final_score = harmonic_score(src_score, tgt_score)

    return {
        "score": round(final_score, 4),
        "src_score": round(src_score, 4),
        "tgt_score": round(tgt_score, 4),
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


def get_thresholds() -> Tuple[float, float]:
    if SIMILARITY_MODE == "fuzzy":
        return FUZZY_EDGE_THRESHOLD, FUZZY_REPRESENTATIVE_THRESHOLD

    if SIMILARITY_MODE == "embedding":
        return EMBEDDING_EDGE_THRESHOLD, EMBEDDING_REPRESENTATIVE_THRESHOLD

    raise ValueError("SIMILARITY_MODE must be either 'fuzzy' or 'embedding'")


def get_cached_score(
    i: int,
    j: int,
    score_cache: Dict[Tuple[int, int], Dict[str, Any]]
) -> Dict[str, Any]:
    
    """
    Retrieves a previously computed similarity score.

    Pairwise similarities are computed only once and
    stored in a cache in order to avoid redundant
    comparisons during:
    - representative selection
    - cluster refinement

    This considerably reduces computational cost.
    """

    if i == j:
        return {
            "is_match": True,
            "score": 1.0,
            "src_score": 1.0,
            "tgt_score": 1.0,
        }

    key = (min(i, j), max(i, j))

    return score_cache.get(
        key,
        {
            "is_match": False,
            "score": 0.0,
            "src_score": 0.0,
            "tgt_score": 0.0,
        }
    )


def choose_representative_item(
    cluster_items: List[Dict[str, Any]],
    score_cache: Dict[Tuple[int, int], Dict[str, Any]]
) -> Dict[str, Any]:
    
    """
    Selects the cluster representative.

    For every triple in the cluster, the average
    similarity to all other cluster members is computed.

    The triple with the highest average similarity
    is selected as the representative because it
    best captures the central semantic meaning
    of the cluster.
    """

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


def compute_pair_score(
    i: int,
    j: int,
    all_items: List[Dict[str, Any]],
    subject_embeddings: np.ndarray | None,
    object_embeddings: np.ndarray | None,
    edge_threshold: float
) -> Dict[str, Any]:        # Computes similarity between two triples.
    triple1 = all_items[i]["triple"]
    triple2 = all_items[j]["triple"]

    if not relations_match(triple1[1], triple2[1]):
        return {
            "is_match": False,
            "score": 0.0,
            "src_score": 0.0,
            "tgt_score": 0.0,
        }

    if SIMILARITY_MODE == "fuzzy":
        result = fuzzy_pair_score(triple1, triple2)

    elif SIMILARITY_MODE == "embedding":
        result = embedding_pair_score(
            i,
            j,
            subject_embeddings,
            object_embeddings
        )

    else:
        raise ValueError("Invalid SIMILARITY_MODE")

    score = result["score"]
    src_score = result["src_score"]
    tgt_score = result["tgt_score"]

    is_match = (
        score >= edge_threshold
        and src_score >= MIN_SUBJECT_SCORE
        and tgt_score >= MIN_OBJECT_SCORE
    )

    return {
        "is_match": is_match,
        "score": round(score, 4),
        "src_score": round(src_score, 4),
        "tgt_score": round(tgt_score, 4),
    }


# =========================
# MAIN
# =========================

def main() -> None:
    edge_threshold, representative_threshold = get_thresholds()

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

    subject_embeddings = None
    object_embeddings = None

    if SIMILARITY_MODE == "embedding":
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        subject_texts = [
            build_embedding_text(item["triple"][0])
            for item in all_items
        ]

        object_texts = [
            build_embedding_text(item["triple"][2])
            for item in all_items
        ]

        print("Encoding subjects...")
        subject_embeddings = model.encode(
            subject_texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        print("Encoding objects...")
        object_embeddings = model.encode(
            object_texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    print(f"Loaded triples: {n}")
    print(f"Similarity mode: {SIMILARITY_MODE}")
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

                result = compute_pair_score(
                    i=i,
                    j=j,
                    all_items=all_items,
                    subject_embeddings=subject_embeddings,
                    object_embeddings=object_embeddings,
                    edge_threshold=edge_threshold
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
            result = get_cached_score(
                representative_id,
                item["global_id"],
                score_cache
            )

            similarity_to_representative = result["score"]

            if similarity_to_representative >= representative_threshold:
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
    print(f"Similarity mode: {SIMILARITY_MODE}")
    print(f"Edge threshold: {edge_threshold}")
    print(f"Representative threshold: {representative_threshold}")
    print(f"Minimum subject score: {MIN_SUBJECT_SCORE}")
    print(f"Minimum object score: {MIN_OBJECT_SCORE}")
    print(f"Minimum pattern frequency: {MIN_PATTERN_FREQUENCY}")
    print(f"Minimum distinct sentences: {MIN_DISTINCT_SENTENCES}")
    print(f"Total triples: {n}")
    print(f"Final recurring patterns found: {final_patterns_count}")


if __name__ == "__main__":
    main()