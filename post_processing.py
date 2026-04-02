import re


class TripletPostProcessor:
    def __init__(self, rels, logger):
        self.rels = rels
        self.logger = logger

        self.rels_set = set(self.rels.tolist())

        self.schema_norm = {
            self.normalize_relation(r): r
            for r in self.rels_set
        }

        raw_rel_mapping = {
            # CAUSES
            "causes": "CAUSES",
            "cause": "CAUSES",
            "caused by": "CAUSES",
            "leads to": "CAUSES",
            "lead to": "CAUSES",
            "results in": "CAUSES",
            "result in": "CAUSES",
            "drives": "CAUSES",
            "induces": "CAUSES",
            "triggers": "CAUSES",
            "contributes to": "CAUSES",

            # AFFECTS
            "affects": "AFFECTS",
            "affect": "AFFECTS",
            "influences": "AFFECTS",
            "influence": "AFFECTS",
            "impacts": "AFFECTS",
            "impact": "AFFECTS",
            "modifies": "AFFECTS",
            "alters": "AFFECTS",
            "limits": "AFFECTS",
            "limit": "AFFECTS",

            # CONVERTED_TO
            "converted to": "CONVERTED_TO",
            "convert to": "CONVERTED_TO",
            "conversion to": "CONVERTED_TO",
            "transformed into": "CONVERTED_TO",
            "transform into": "CONVERTED_TO",
            "changed to": "CONVERTED_TO",
            "change to": "CONVERTED_TO",
            "turned into": "CONVERTED_TO",
            "converted into": "CONVERTED_TO",

            # LOCATED_IN
            "located in": "LOCATED_IN",
            "within": "LOCATED_IN",
            "inside": "LOCATED_IN",
            "in the region of": "LOCATED_IN",
            "in the area of": "LOCATED_IN",
            "found in": "LOCATED_IN",
            "present in": "LOCATED_IN",

            # OCCURS_DURING
            "during": "OCCURS_DURING",
            "over": "OCCURS_DURING",
            "between": "OCCURS_DURING",
            "over the period": "OCCURS_DURING",
            "within the period": "OCCURS_DURING",

            # INCREASES
            "increase": "INCREASES",
            "increases": "INCREASES",
            "increasing": "INCREASES",
            "grow": "INCREASES",
            "grows": "INCREASES",
            "growing": "INCREASES",
            "grew": "INCREASES",
            "growth": "INCREASES",
            "rise": "INCREASES",
            "rises": "INCREASES",
            "rising": "INCREASES",
            "rose": "INCREASES",
            "expand": "INCREASES",
            "expands": "INCREASES",
            "expansion": "INCREASES",
            "expanded": "INCREASES",

            # INCREASED_BY
            "increased by": "INCREASED_BY",
            "increase by": "INCREASED_BY",
            "rose by": "INCREASED_BY",
            "growth of": "INCREASED_BY",
            "increase of": "INCREASED_BY",

            # DECREASES
            "decrease": "DECREASES",
            "decreases": "DECREASES",
            "decreasing": "DECREASES",
            "decline": "DECREASES",
            "declines": "DECREASES",
            "declining": "DECREASES",
            "declined": "DECREASES",
            "reduce": "DECREASES",
            "reduces": "DECREASES",
            "reduction": "DECREASES",
            "loss": "DECREASES",
            "shrink": "DECREASES",
            "shrinks": "DECREASES",

            # DECREASED_BY
            "decreased by": "DECREASED_BY",
            "decrease by": "DECREASED_BY",
            "declined by": "DECREASED_BY",
            "reduced by": "DECREASED_BY",
            "reduction of": "DECREASED_BY",
            "loss of": "DECREASED_BY",

            # FROM_TO
            "ranged from": "FROM_TO",
            "changed from": "FROM_TO",
            "varied from": "FROM_TO",
        }

        self.rel_mapping = {
            self.normalize_relation(k): v
            for k, v in raw_rel_mapping.items()
        }

    def normalize_relation(self, rel: str) -> str:
        rel = rel.lower().strip()
        rel = rel.replace("_", " ")
        rel = re.sub(r"\s+", " ", rel)
        return rel

    def handle_ambiguous_cases(self, rel_norm, rel_tri):
        _h, _r, _t = rel_tri
        t = str(_t).lower().strip()

        if rel_norm == "in":
            temporal_markers = [
                "year", "years", "period", "century", "season",
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december",
                "spring", "summer", "autumn", "fall", "winter"
            ]

            if (
                any(x in t for x in temporal_markers)
                or re.search(r"\b(18|19|20)\d{2}\b", t)
                or re.search(r"\b(18|19|20)\d{2}\s*[-–]\s*(18|19|20)\d{2}\b", t)
            ):
                return "OCCURS_DURING"

            return "LOCATED_IN"

        if "from" in rel_norm or "to" in rel_norm:
            has_explicit_transition = (
                re.search(r"\bfrom\b.*\bto\b", t)
                or re.search(r"\b\d+(\.\d+)?\s*(%|km²|ha|hectares)?\s*(to|[-–])\s*\d+(\.\d+)?\s*(%|km²|ha|hectares)?\b", t)
            )

            if has_explicit_transition:
                return "FROM_TO"

            return None

        return None

    def looks_temporal(self, text: str) -> bool:
        t = text.lower().strip()
        temporal_words = [
            "year", "years", "period", "season", "month", "months",
            "spring", "summer", "autumn", "fall", "winter",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "before", "after", "during", "since"
        ]
        return (
            any(w in t for w in temporal_words)
            or re.search(r"\b(18|19|20)\d{2}\b", t) is not None
            or re.search(r"\b(18|19|20)\d{2}\s*[-–]\s*(18|19|20)\d{2}\b", t) is not None
            or "second period" in t
            or "first period" in t
        )

    def looks_quantitative(self, text: str) -> bool:
        t = text.lower().strip()
        return (
            re.search(r"\b\d+(\.\d+)?\b", t) is not None
            or "%" in t
            or "km²" in t
            or "km2" in t
            or "ha" in t
            or "hectare" in t
            or "hectares" in t
            or "half" in t
            or "quarter" in t
        )

    def looks_transition(self, text: str) -> bool:
        t = text.lower().strip()
        return (
            re.search(r"\bfrom\b.*\bto\b", t) is not None
            or re.search(r"\b\d+(\.\d+)?\s*(%|km²|km2|ha|hectares)?\s*(to|[-–])\s*\d+(\.\d+)?\s*(%|km²|km2|ha|hectares)?\b", t) is not None
        )

    def is_valid_triplet(self, head: str, rel: str, tail: str) -> bool:
        t = tail.lower().strip()

        if rel == "OCCURS_DURING":
            return self.looks_temporal(t)

        if rel in {"INCREASED_BY", "DECREASED_BY"}:
            if not self.looks_quantitative(t):
                return False
            bad_quant_targets = ["scarce", "observed loss", "fertility", "size"]
            if any(x == t for x in bad_quant_targets):
                return False
            return True

        if rel == "FROM_TO":
            return self.looks_transition(t)

        if rel == "CONVERTED_TO":
            bad_convert_targets = ["irreversible", "scarce", "defunct", "marginal"]
            if t in bad_convert_targets:
                return False
            return True

        if rel == "LOCATED_IN":
            if self.looks_temporal(t):
                return False
            if self.looks_quantitative(t):
                return False
            return True

        return True

    def get_simi_rel_by_relcanon(self, rel, raw_sent, rel_tri):
        rel_norm = self.normalize_relation(rel)

        if rel_norm in self.schema_norm:
            return self.schema_norm[rel_norm]

        for key in sorted(self.rel_mapping.keys(), key=len, reverse=True):
            if key in rel_norm:
                mapped = self.rel_mapping[key]
                if mapped in self.rels_set:
                    self.logger.info(f"🔁 Mapped relation: {rel} -> {mapped}")
                    return mapped

        special = self.handle_ambiguous_cases(rel_norm, rel_tri)
        if special is not None:
            return special

        self.logger.warning(
            f"⚠️ Relation out of schema discarded: rel={rel} | norm={rel_norm} | sentence={raw_sent}"
        )
        return None

    def process_pair(self, pair, sentence):
        try:
            tri_, tri_text, entailment = pair   # tri_text currently unused
            _h, _r, _t = tri_
        except Exception:
            return None

        simi_rel = self.get_simi_rel_by_relcanon(_r, sentence, tri_)

        if simi_rel is None:
            return None

        if not self.is_valid_triplet(_h, simi_rel, _t):
            self.logger.warning(
                f"⚠️ Invalid semantic triplet discarded: [{_h}, {simi_rel}, {_t}]"
            )
            return None

        final_tri = [_h, simi_rel, _t]
        return final_tri, entailment