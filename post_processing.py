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
            "increased by": "INCREASES",
            "increase by": "INCREASES",
            "rose by": "INCREASES",
            "growth of": "INCREASES",
            "increase of": "INCREASES",

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
            "decreased by": "DECREASES",
            "decrease by": "DECREASES",
            "declined by": "DECREASES",
            "reduced by": "DECREASES",
            "reduction of": "DECREASES",
            "loss of": "DECREASES",

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
        t = self.clean_text_field(text).lower()

        temporal_words = [
            "year", "years", "period", "periods", "season", "seasons",
            "month", "months", "decade", "decades", "century", "centuries",
            "spring", "summer", "autumn", "fall", "winter",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
            "before", "after", "during", "since", "until", "over",
            "annual", "yearly", "historical", "future"
        ]

        return (
            any(w in t for w in temporal_words)

            # anni singoli: 1990
            or re.search(r"\b(18|19|20)\d{2}\b", t) is not None

            # decenni: 1970s, 1980s
            or re.search(r"\b(18|19|20)\d{2}s\b", t) is not None

            # intervalli tipo 1990-2010 / 1990–2010
            or re.search(r"\b(18|19|20)\d{2}\s*[-–]\s*(18|19|20)\d{2}\b", t) is not None

            # from X to Y / between X and Y
            or re.search(r"\bfrom\b.+\bto\b", t) is not None
            or re.search(r"\bbetween\b.+\band\b", t) is not None

            # by + anno
            or re.search(r"\bby\s+(18|19|20)\d{2}\b", t) is not None

            # in the 1970s / during the 1980s
            or re.search(r"\b(in|during|after|before)\s+the\s+(18|19|20)\d{2}s\b", t) is not None

            # early/late + season/period/year
            or re.search(r"\b(early|late)\s+(growing\s+)?(season|period|year)\b", t) is not None

            # after/before + event nominale con parola temporale implicita
            or re.search(r"\b(after|before|during)\s+.+", t) is not None
        )

    
    def looks_like_state_or_quality(self, text: str) -> bool:
        t = self.clean_text_field(text).lower()

        # una sola parola e molto probabilmente aggettivo/stato
        bad_single_words = {
            "irreversible", "scarce", "scarcity", "defunct",
            "marginal", "limited", "adequate", "high", "low"
        }
        if t in bad_single_words:
            return True

        # target introdotti da spiegazioni/causalità
        if t.startswith(("due to ", "because of ")):
            return True

        return False
    
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
    
    def looks_like_human_group(self, text: str) -> bool:
        t = self.clean_text_field(text).lower()

        human_markers = [
            "population", "people", "farmers", "households",
            "rural population", "urban population", "community", "communities"
        ]

        return any(w in t for w in human_markers)
    


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

        if r == "OCCURS_DURING":
            h_temporal = self.looks_temporal(h)
            t_temporal = self.looks_temporal(t)

            if not t_temporal:
                return False

            if h_temporal:
                return False

            return True

        if r == "CONVERTED_TO":
            if self.looks_temporal(t):
                return False
            if self.looks_quantitative(t):
                return False
            if self.looks_like_state_or_quality(t):
                return False
            return True
        
        if r == "DOMINATES":
            if self.looks_temporal(t):
                return False
            if self.looks_quantitative(t):
                return False
            if self.looks_like_state_or_quality(t):
                return False
            if self.looks_like_human_group(t):
                return False
            return True

        # location: più permissivo di prima
        if r == "LOCATED_IN":
            if self.looks_temporal(t):
                return False
            if self.looks_quantitative(t):
                return False
            return True

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
    
    def normalize_occurs_during_direction(self, head: str, tail: str):
        h = self.clean_text_field(head)
        t = self.clean_text_field(tail)

        h_temporal = self.looks_temporal(h)
        t_temporal = self.looks_temporal(t)

        # Caso invertito: tempo -> evento
        if h_temporal and not t_temporal:
            self.logger.info(f"🔁 Swapped OCCURS_DURING arguments: [{h}] <-> [{t}]")
            return t, h

        return h, t

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

        if simi_rel == "OCCURS_DURING":
            _h, _t = self.normalize_occurs_during_direction(_h, _t)

        if not self.is_valid_triplet(_h, simi_rel, _t):
            self.logger.warning(
                f"⚠️ Invalid semantic triplet discarded: [{_h}, {simi_rel}, {_t}]"
            )
            return None
        
        final_tri = [_h, simi_rel, _t]
        return final_tri, entailment