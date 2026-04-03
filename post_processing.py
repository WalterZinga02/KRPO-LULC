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
            "impacts": "CAUSES",
            "impact": "CAUSES",
            "influences": "CAUSES",
            "influence": "CAUSES",

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

            # DOMINATES
            "dominates": "DOMINATES",
            "dominate": "DOMINATES",
            "dominated": "DOMINATES",
            "dominant": "DOMINATES",
            "predominant": "DOMINATES",
            "prevalent": "DOMINATES",
            "is prevalent in": "DOMINATES",
        }

        self.rel_mapping = {
            self.normalize_relation(k): v
            for k, v in raw_rel_mapping.items()
        }

    def clean_text_field(self, value) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def is_placeholder(self, text: str) -> bool:
        t = text.lower().strip()

        placeholder_values = {
            "", ".", ",", ";", ":", "-", "_", "--", "---",
            "none", "null", "n/a", "na", "unknown", "unspecified",
            "not available", "not specified", "missing", "blank"
        }

        if t in placeholder_values:
            return True

        if re.fullmatch(r"[\W_]+", t):
            return True

        return False

    def normalize_relation(self, rel: str) -> str:
        rel = "" if rel is None else str(rel)
        rel = rel.lower().strip()
        rel = rel.replace("_", " ")
        rel = re.sub(r"\s+", " ", rel)
        return rel

    def handle_ambiguous_cases(self, rel_norm, rel_tri):
        _h, _r, _t = rel_tri
        t = self.clean_text_field(_t).lower()

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

        return None

    def looks_temporal(self, text: str) -> bool:
        t = text.lower().strip()
        temporal_words = [
            "year", "years", "period", "season", "month", "months",
            "spring", "summer", "autumn", "fall", "winter",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "before", "after", "during", "since",
            "by 2050", "each year", "year-round"
        ]
        return (
            any(w in t for w in temporal_words)
            or re.search(r"\b(18|19|20)\d{2}\b", t) is not None
            or re.search(r"\b(18|19|20)\d{2}\s*[-–]\s*(18|19|20)\d{2}\b", t) is not None
            or "second period" in t
            or "first period" in t
            or t.startswith("from ")
            or t.startswith("between ")
            or t.startswith("in ")
            or t.startswith("by ")
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
            or "million" in t
            or "less than" in t
            or "more than" in t
        )

    def normalize_change_relation(self, rel: str, tail: str) -> str:
        t = self.clean_text_field(tail)

        if rel == "INCREASES" and self.looks_quantitative(t):
            return "INCREASED_BY"

        if rel == "DECREASES" and self.looks_quantitative(t):
            return "DECREASED_BY"

        return rel

    def is_valid_triplet(self, head: str, rel: str, tail: str) -> bool:
        h = self.clean_text_field(head)
        r = self.clean_text_field(rel)
        t = self.clean_text_field(tail)

        # resta sempre rigido su placeholder / campi rotti
        if self.is_placeholder(h) or self.is_placeholder(r) or self.is_placeholder(t):
            return False

        if len(h) < 2 or len(r) < 2 or len(t) < 2:
            return False

        if h.lower() == t.lower():
            return False

        # OCCURS_DURING: ancora abbastanza rigido
        if r == "OCCURS_DURING":
            return self.looks_temporal(t)

        # quantità: ancora abbastanza rigido
        if r in {"INCREASED_BY", "DECREASED_BY"}:
            return self.looks_quantitative(t)

        # conversione: diventiamo più permissivi, scartiamo solo placeholder evidenti
        if r == "CONVERTED_TO":
            bad_convert_targets = {"irreversible", "scarce"}
            return t.lower() not in bad_convert_targets

        # dominanza: abbastanza permissivo
        if r == "DOMINATES":
            if self.looks_temporal(t):
                return False
            if self.looks_quantitative(t):
                return False
            return True

        # location: più permissivo di prima
        if r == "LOCATED_IN":
            if self.looks_temporal(t):
                return False
            if self.looks_quantitative(t):
                return False

            bad_location_targets = {
                "agricultural cycle",
                "irrigation",
                "communal access",
                "cover in 1990"
            }
            return t.lower() not in bad_location_targets

        # increases/decreases: molto permissivo, purché non sia placeholder
        if r in {"INCREASES", "DECREASES"}:
            return True

        # causes: permissivo
        if r == "CAUSES":
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
            tri_, tri_text, entailment = pair
            _h, _r, _t = tri_
        except Exception:
            return None

        _h = self.clean_text_field(_h)
        _r = self.clean_text_field(_r)
        _t = self.clean_text_field(_t)

        simi_rel = self.get_simi_rel_by_relcanon(_r, sentence, (_h, _r, _t))

        if simi_rel is None:
            return None

        # normalizza increases/decreases -> *_BY quando il tail è quantitativo
        simi_rel = self.normalize_change_relation(simi_rel, _t)

        if not self.is_valid_triplet(_h, simi_rel, _t):
            self.logger.warning(
                f"⚠️ Invalid semantic triplet discarded: [{_h}, {simi_rel}, {_t}]"
            )
            return None

        final_tri = [_h, simi_rel, _t]
        return final_tri, entailment