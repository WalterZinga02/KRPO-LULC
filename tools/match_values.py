from rapidfuzz import fuzz

def entity_score(a: str, b: str) -> float:
    """
    Blend three metrics:
    - ratio: strict full-string similarity (length-sensitive)
    - token_sort_ratio: handles word reordering
    - partial_ratio: handles substrings, but weighted down to avoid false 1.0s
    """
    r  = fuzz.ratio(a, b) / 100.0
    ts = fuzz.token_sort_ratio(a, b) / 100.0
    pr = fuzz.partial_ratio(a, b) / 100.0
    print(r,ts,pr)
    return round(0.4 * ts + 0.4 * r + 0.2 * pr, 4)



def match_triples(
    triple1: tuple[str, str, str],
    triple2: tuple[str, str, str],
    entity_threshold: float = 0.6,
) -> float | None:
    src1, rel1, tgt1 = triple1
    src2, rel2, tgt2 = triple2

    if rel1.strip().upper() != rel2.strip().upper():
        return None

    src_score = entity_score(src1.lower(), src2.lower())
    tgt_score = entity_score(tgt1.lower(), tgt2.lower())

    if src_score < entity_threshold or tgt_score < entity_threshold:
        return None

    return round((src_score + tgt_score) / 2, 4)



if __name__ == "__main__":
    pairs = [

        (
            ("Walter", "GOES_TO", "Fishing"),
            ("Walterino", "GOES_TO", "Fishing"),
            "Person name partial + location partial",
        ),
        (
           ('pasture expansion', 'ASSOCIATED_WITH', 'soy expansion'),
          ('pasture', 'ASSOCIATED_WITH', 'soy expansion'),
            "Person name partial + location partial",
        ),
        (
            ('natural land cover types', 'DOMINATES', 'study region'),
            ('natural land cover types', 'DOMINATES', 'almost three quarters (73%) of the study region'),
            ""
        ),
        (
            ('natural land cover types', 'DOMINATES', 'study region'),
            ('natural land cover types', 'DOMINATES', 'almost three quarters (73%) of the study region'),
            ""
        )
    ]

    threshold = 0.2
    print(f"Entity threshold: {threshold}\n")

    for t1, t2, desc in pairs:
        score = match_triples(t1, t2, entity_threshold=threshold)
        result = f"{score:.4f}" if score is not None else "None (filtered)"
        print(f"── {desc}")
        print(f"   Triple A : {t1}")
        print(f"   Triple B : {t2}")
        print(f"   Score    : {result}")
        print()