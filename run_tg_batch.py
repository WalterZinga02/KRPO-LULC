import json
import os
import re
import time
import pandas as pd

import numpy as np
import ast
import logging
from tqdm import tqdm
from model_utils.llms import LLMInvoker, UTF8FileHandler, text_relcanon
from tools.replace_h_t_4reldef import replace_entities
from prompts.optimizer_prompts import BACKWARD_prompt_EVAL, BACKWARD_prompt_PRED, OPTIMIZER_EXAMPLE_TEMPLATE, OPTIMIZER_prompt
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken
from post_processing import TripletPostProcessor

# Settings
choose_llm = "gpt-4o-mini"
chose_dataset = "lulc_test"
save_dir = "outputs"
log_rm = "test1_"

os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), exist_ok=True)

main_logger = logging.getLogger('main')
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", f"{save_dir}{log_rm}{chose_dataset}_inout.log")
main_handler = UTF8FileHandler(log_file, mode='w', encoding='utf-8')
main_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
main_logger.addHandler(main_handler)
main_logger.setLevel(logging.INFO)
enc = tiktoken.get_encoding("cl100k_base")


def read_sentences_from_file(read_file):
    sentences = []
    with open(read_file, 'r', encoding='utf-8') as file:
        for line in file:
            sentences.append(line.strip())
    return sentences


oie_prompt = """Your task is to transform the given text into a semantic graph in the form of a list of triples.

Return ONLY a Python-style list of triples in the following format:
[[subject, predicate, object], [subject, predicate, object], ...]

Do not include any explanation or additional text.

Guidelines:
Extract all meaningful triples from the sentence.
Focus especially on land use, land cover, environmental processes, and their changes (LULC).
Each triple must be clear and semantically complete.

Preferred relation types include:
CAUSES, AFFECTS, CONVERTED_TO, LOCATED_IN, OCCURS_DURING, INCREASES, INCREASED_BY, DECREASES, DECREASED_BY, FROM_TO.

These relation types are preferred when they naturally fit the sentence, but do not force them if they reduce clarity or correctness.
If not possible, use the most concise predicate that preserves the meaning.

Avoid vague predicates such as "is", "has", "related to", or "associated with".
Prefer informative and specific relations.

Output must be:

a valid Python list
no explanations
no newline characters or formatting symbols
no additional text before or after the list
"""
suffix_prompt = """
Here are some examples:
{few_shot_examples}

Now please extract triplets from the following text.
Text: {input_text}
Triplets: """


class KG4:
    def __init__(self, this_dataset, this_llm):
        llm_loger = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"./logs/{save_dir}{log_rm}{chose_dataset}_log.log"
        )
        self.llm = LLMInvoker(this_llm, llm_loger)
        self.data_sentences = read_sentences_from_file(f"./datasets/{this_dataset}.txt")

        # --- prompt txt ---
        self.extract_few_shot_examples = open(
            f"./prompts/few_shot_examples/{this_dataset}/oie_few_shot_examples.txt",
            encoding='utf-8'
        ).read()

        self.triplet2text_prompt_template = open(
            "./prompts/main_prompt/triplet2text.txt",
            encoding='utf-8'
        ).read()

        self.nli_few_shot_examples = open(
            "./prompts/few_shot_examples/NLI.txt",
            encoding='utf-8'
        ).read()

        self.nli_prompt_template = open(
            "./prompts/main_prompt/NLI.txt",
            encoding='utf-8'
        ).read()

        self.backward_1_template = BACKWARD_prompt_EVAL
        self.backward_2_template = BACKWARD_prompt_PRED
        self.optimizer_examples = OPTIMIZER_EXAMPLE_TEMPLATE
        self.optimizer_template = OPTIMIZER_prompt

        self.rel_simi_choice_template = open(
            "./prompts/main_prompt/rel_simi_choice.txt",
            encoding='utf-8'
        ).read()

        local_rels = pd.read_csv(
            f"./schemas/{this_dataset}_schema.csv",
            header=None,
            names=["name", "schema"]
        )

        self.rels = local_rels["name"].to_numpy()
        self.rel_schemas = local_rels["schema"].to_numpy()


    # MODIFICA: normalizzazione robusta per relazioni
    def normalize_relation(self, rel: str) -> str:
        rel = rel.lower().strip()
        rel = rel.replace("_", " ")
        rel = re.sub(r"\s+", " ", rel)
        return rel

    # MODIFICA: gestione più conservativa dei casi ambigui
    def handle_ambiguous_cases(self, rel_norm, rel_tri):
        _h, _r, _t = rel_tri
        t = str(_t).lower().strip()

        # Caso "in": distinguere tempo da luogo
        if rel_norm == "in":
            temporal_markers = [
                "year", "years", "period", "century", "season",
                "january", "february", "march", "april", "may", "june",
                "july", "august", "september", "october", "november", "december",
                "spring", "summer", "autumn", "fall", "winter"
            ]

            # pattern tipo 1984, 2001, 1950-1998, ecc.
            if (
                any(x in t for x in temporal_markers)
                or re.search(r"\b(18|19|20)\d{2}\b", t)
                or re.search(r"\b(18|19|20)\d{2}\s*[-–]\s*(18|19|20)\d{2}\b", t)
            ):
                return "OCCURS_DURING"

            return "LOCATED_IN"

        # Caso FROM_TO: accettarlo solo se l'oggetto contiene davvero una transizione
        if "from" in rel_norm or "to" in rel_norm:
            has_explicit_transition = (
                re.search(r"\bfrom\b.*\bto\b", t)
                or re.search(r"\b\d+(\.\d+)?\s*(%|km²|ha|hectares)?\s*(to|[-–])\s*\d+(\.\d+)?\s*(%|km²|ha|hectares)?\b", t)
            )

            if has_explicit_transition:
                return "FROM_TO"

            # se non c'è una vera transizione, non mappare
            return None

        return None
    
    # MODIFICA: riconosce valori temporali
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
    
    # MODIFICA: riconosce quantità/misure
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
    
    # MODIFICA: riconosce vere transizioni
    def looks_transition(self, text: str) -> bool:
        t = text.lower().strip()
        return (
            re.search(r"\bfrom\b.*\bto\b", t) is not None
            or re.search(r"\b\d+(\.\d+)?\s*(%|km²|km2|ha|hectares)?\s*(to|[-–])\s*\d+(\.\d+)?\s*(%|km²|km2|ha|hectares)?\b", t) is not None
        )
    
    # MODIFICA: filtro semantico finale leggero
    def is_valid_triplet(self, head: str, rel: str, tail: str) -> bool:
        h = head.lower().strip()
        t = tail.lower().strip()

        # relazioni temporali
        if rel == "OCCURS_DURING":
            return self.looks_temporal(t)

        # relazioni quantitative
        if rel in {"INCREASED_BY", "DECREASED_BY"}:
            if not self.looks_quantitative(t):
                return False
            # scarta casi palesemente sbagliati
            bad_quant_targets = ["scarce", "observed loss", "fertility", "size"]
            if any(x == t for x in bad_quant_targets):
                return False
            return True

        # transizioni
        if rel == "FROM_TO":
            return self.looks_transition(t)

        # conversioni
        if rel == "CONVERTED_TO":
            # scarta aggettivi/stati non entità
            bad_convert_targets = ["irreversible", "scarce", "defunct", "marginal"]
            if t in bad_convert_targets:
                return False
            return True

        # location: evita valori chiaramente temporali o quantitativi
        if rel == "LOCATED_IN":
            if self.looks_temporal(t):
                return False
            if self.looks_quantitative(t):
                return False
            return True

        # AFFECTS / CAUSES / INCREASES / DECREASES di default
        return True

    # MODIFICA IMPORTANTE: estrazione RAW, senza filtraggio finale schema
    # Questo mantiene KRPO coerente: extraction/NLI/optimization lavorano sui raw triplets.
    def sentence_extract_triplets(self, sentence, _oie_prompt):
        main_logger.info(f"\nextracting...\n{'-' * 50}\n")

        extract_prompt = _oie_prompt + suffix_prompt.format_map({
            'few_shot_examples': self.extract_few_shot_examples,
            'input_text': sentence
        })

        for retry_count in range(3):
            extract_res = None

            for llm_retry in range(3):
                main_logger.info(f"\n{'-' * 50}\nExtract sentence:\n{sentence}")
                extract_res = self.llm.llm_chat_response(extract_prompt)
                main_logger.info(f"\nExtract Response:\n{extract_res}\n{'-' * 50}\n")

                if extract_res is not None:
                    break
                time.sleep(1)

            if extract_res is None:
                main_logger.error("❌ LLM - None, max tries reached")
                return []

            try:
                extract_res = extract_res.strip()

                # Supporta sia {"triplets": [...]} sia lista pura [...]
                if extract_res.startswith("{"):
                    parsed = json.loads(extract_res)
                    extract_res_list = parsed.get("triplets", [])
                else:
                    if not (extract_res.startswith('[') and extract_res.endswith(']')):
                        match = re.search(r"\[.*\]", extract_res, re.DOTALL)
                        if not match:
                            raise ValueError("No valid list found in response")
                        extract_res = match.group(0)

                    extract_res_list = ast.literal_eval(extract_res)

            except Exception as e:
                main_logger.error(f"❌ extract res parse fail: {e} | raw={repr(extract_res)}")
                main_logger.warning(f"⚠️ Parse error {retry_count + 1}: {e}")
                time.sleep(1)
                continue

            try:
                valid_triplets = []
                for tri in extract_res_list:
                    if not isinstance(tri, (list, tuple)) or len(tri) != 3:
                        continue

                    head, relation, tail = tri
                    if not all(isinstance(x, str) for x in (head, relation, tail)):
                        continue

                    # MODIFICA: qui NON facciamo mapping/filtro schema,
                    # lasciamo passare le raw triplets per la fase KRPO.
                    valid_triplets.append([head, relation, tail])

            except Exception as e:
                main_logger.warning(f"⚠️ Triplet validation error: {e}")
                continue

            main_logger.info(
                f"### currentProcessingSentence: {repr(sentence)}\nextractTripletResults {valid_triplets}"
            )
            return valid_triplets

        main_logger.error("❌ Maximum retries reached, no valid triplets")
        return []

    def triplet_to_text(self, single_triplet):
        main_logger.info(f"\nrestoringASingleTriple...\n{'-' * 50}\n")
        tri2text_prompt = self.triplet2text_prompt_template.format_map({"input_triplet": single_triplet})
        main_logger.info(f"\n{'-' * 50}\nTri2Txt single_triplet:\n{single_triplet}")
        tri2text_res = self.llm.llm_chat_response(tri2text_prompt)
        main_logger.info(f"\nTri2Txt Response:\n{tri2text_res}\n{'-' * 50}\n")
        return tri2text_res

    def nli_single(self, raw_sentence, sent_by_triplet):
        main_logger.info(f"\nNLI...\n{'-' * 50}\n")
        nli_prompt = self.nli_prompt_template.format_map({
            "few_shot_examples": self.nli_few_shot_examples,
            "raw_sentence": raw_sentence,
            "kg_text": sent_by_triplet
        })

        nli_result = None
        while nli_result is None:
            response = self.llm.llm_chat_response(nli_prompt)
            if response is not None and len(enc.encode(response)) <= 4196:
                nli_result = response

        main_logger.info(
            f"\n{'-' * 50}\nNLI sent:\n{raw_sentence=}\n{sent_by_triplet=}\nNLI Response:\n{nli_result}\n{'-' * 50}\n"
        )

        try:
            json_match = re.search(r'```json\s*(.*?)```', nli_result, flags=re.DOTALL)
            if json_match:
                nli_result = json_match.group(1).strip()
            nli_result = re.sub(r',\s*"reasoning"\s*:\s*".*$', '}', nli_result, flags=re.DOTALL)
            nli_res = json.loads(nli_result.replace('\n', ''))
            return nli_res
        except (ValueError, SyntaxError) as e:
            main_logger.error(f"❌ Single sentence by triple NLI result {repr(nli_result)} parseFailed {e}")
            return self.nli_single(raw_sentence, sent_by_triplet)

    # MODIFICA IMPORTANTE: mapping/filtro schema finale
    # Questo viene usato solo nel post-processing, non durante l'estrazione raw.
    def get_simi_rel_by_relcanon(self, rel, relation_text, raw_sent, rel_tri):
        rel_norm = self.normalize_relation(rel)

        # match diretto sullo schema, robusto a maiuscole/underscore/spazi
        if rel_norm in self.schema_norm:
            return self.schema_norm[rel_norm]

        # mapping sinonimi/varianti
        for key in sorted(self.rel_mapping.keys(), key=len, reverse=True):
            if key in rel_norm:
                mapped = self.rel_mapping[key]
                if mapped in self.rels_set:
                    main_logger.info(f"🔁 Mapped relation: {rel} -> {mapped}")
                    return mapped

        # gestione casi ambigui
        special = self.handle_ambiguous_cases(rel_norm, rel_tri)
        if special is not None:
            return special

        main_logger.warning(
            f"⚠️ Relation out of schema discarded: rel={rel} | norm={rel_norm} | sentence={raw_sent}"
        )
        return None

    # Lascio il metodo, anche se ora non stiamo espandendo lo schema
    def add_rel_schema(self, rel, rel_des):
        self.rels = np.append(self.rels, rel)
        self.rel_schemas = np.append(self.rel_schemas, rel_des)

    def extract_and_nli_eval(self, one_sent, extrac_prompt):
        got_triplets = self.sentence_extract_triplets(one_sent, extrac_prompt)
        if len(got_triplets) == 0:
            return 0, [], []

        score = 0.0
        eval_pairs = []

        def process_triplet(ttt):
            ttt_text = self.triplet_to_text(ttt)
            ttt_nli_res = self.nli_single(one_sent, ttt_text)
            ttt_entailment = ttt_nli_res.get("label", "")
            return ttt, ttt_text, ttt_entailment

        max_workers = min(8, len(got_triplets))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_triplet, tt) for tt in got_triplets]
            for future in as_completed(futures):
                tt, tt_text, tt_entailment = future.result()
                eval_pairs.append((tt, tt_text, tt_entailment))
                if tt_entailment == "entailment":
                    score += 1.0
                elif tt_entailment == "contradiction":
                    score -= 0.5

        this_score = score / len(got_triplets)
        return this_score, eval_pairs, got_triplets

    def batch_extract_and_nli_eval(self, batch_sent, ext_prompt):
        b_score, b_eval_pairs, b_got_triplets = [], [], []
        for _id, _sent in enumerate(batch_sent):
            print(f"Extract & NLI: {_id}/{len(batch_sent)}")
            _score, _ev_pair, _triplets = self.extract_and_nli_eval(_sent, ext_prompt)
            b_score.append(_score)
            b_eval_pairs.append(_ev_pair)
            b_got_triplets.append(_triplets)
        return b_score, b_eval_pairs, b_got_triplets

    def backward_1(self, res_extract, res_eval, res_score):
        _eval_strs = [f"\t{str(_tri)}: {_entailment}" for _tri, _text, _entailment in res_eval]
        _eval_str = "\n".join(_eval_strs)
        _eval_str += f"\nscore: {res_score:.2f}"

        optimize_1_prompt = self.backward_1_template.replace(
            "__predict_triplets__", str(res_extract)
        ).replace(
            "__evaluate_result__", _eval_str
        )

        main_logger.info(f"\nbackward_1...\n{'-' * 50}\n")
        main_logger.info(f"\n{'-' * 50}\nBackward_1:\n{res_extract=}\n{_eval_str=}\n")

        optimize_1 = None
        while optimize_1 is None:
            optimize_1 = self.llm.llm_chat_response(optimize_1_prompt)

        main_logger.info(f"\nBackward1 Response:\n{optimize_1}\n{'-' * 50}\n")
        return optimize_1

    def backward_2(self, input_sent, res_extract, backward1):
        global oie_prompt

        optimize_2_prompt = self.backward_2_template.replace(
            "__sys_prompt__", oie_prompt
        ).replace(
            "__input_sentence__", input_sent
        ).replace(
            "__extract_response__", str(res_extract)
        ).replace(
            "__backward_response__", backward1
        )

        main_logger.info(f"\n{'-' * 50}\nBackward_2:\n{input_sent=}\n{res_extract=}\n{backward1=}\n")
        main_logger.info(f"\nbackward_2...\n{'-' * 50}\n")

        optimize_2 = None
        while optimize_2 is None:
            optimize_2 = self.llm.llm_chat_response(optimize_2_prompt)

        main_logger.info(f"\nBackward2 Response:\n{optimize_2}\n{'-' * 50}\n")
        return optimize_2

    def backward(self, input_sent, res_extract, res_eval, res_score):
        opt_1 = self.backward_1(res_extract, res_eval, res_score)
        opt_2 = self.backward_2(input_sent, res_extract, opt_1)
        return opt_2

    def optimizer(self, one_in, one_out, one_bak):
        global oie_prompt
        _sample_example = self.optimizer_examples.replace(
            "__sys_prompt__", oie_prompt
        ).replace(
            "__input_sentence__", one_in
        ).replace(
            "__extract_response__", str(one_out)
        ).replace(
            "__feedback__", one_bak
        )

        optimizer_info = self.optimizer_template.replace(
            "__sys_prompt__", oie_prompt
        ).replace(
            "__update_insert_examples__", _sample_example
        )

        main_logger.info(f"\nOptimizer update prompt...\n{'-' * 50}\n")
        res_optimize = self.llm.llm_chat_response(optimizer_info)
        return res_optimize

    def optimizer_batch(self, batch_backward):
        global oie_prompt
        infos = []

        for b_in, b_out, b_bak, _ in batch_backward:
            _sample_example = self.optimizer_examples.replace(
                "__sys_prompt__", oie_prompt
            ).replace(
                "__input_sentence__", b_in
            ).replace(
                "__extract_response__", str(b_out)
            ).replace(
                "__feedback__", b_bak
            )
            infos.append(_sample_example)

        info_placeholder = "\n\n".join(infos)

        optimizer_info = self.optimizer_template.replace(
            "__sys_prompt__", oie_prompt
        ).replace(
            "__update_insert_examples__", info_placeholder
        )

        res_optimize = self.llm.llm_chat_response(optimizer_info)
        return res_optimize

def batch_iter(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield [(i + j, sent) for j, sent in enumerate(data[i:i + batch_size])]


def main(dataset, llm_model, output_paths):
    kg4 = KG4(dataset, llm_model)
    global oie_prompt
    batch_size = 5
    post_processor = TripletPostProcessor(kg4.rels, kg4.rel_schemas, main_logger)

    with open(output_paths["raw_triplets_path"], 'w', encoding='utf-8') as out_raw, \
         open(output_paths["entailment_tris_path"], 'w', encoding='utf-8') as out_en, \
         open(output_paths["final_triplets_path"], 'w', encoding='utf-8') as out_f:

        for b_idx, batch in enumerate(
            tqdm(
                batch_iter(kg4.data_sentences, batch_size),
                total=(len(kg4.data_sentences) + batch_size - 1) // batch_size
            )
        ):
            print(f"\n========== BATCH {b_idx + 1} ==========")

            batch_backward_data = []
            batch_sents = []
            batch_score, batch_eval_pairs, batch_extract_triplets = [], [], []

            for sid, sent in batch:
                print(f"\n--- Sentence ID: {sid} ---")
                print("Extract & NLI ...")

                one_sent_score, eval_pairs, extract_triplets = kg4.extract_and_nli_eval(sent, oie_prompt)

                # Debug essenziale
                print("INITIAL TRIPLETS:")
                print(extract_triplets)
                print("INITIAL SCORE:")
                print(one_sent_score)

                batch_score.append(one_sent_score)
                batch_eval_pairs.append(eval_pairs)
                batch_extract_triplets.append(extract_triplets)

                print("Backward ...")
                one_backward = kg4.backward(sent, extract_triplets, eval_pairs, one_sent_score)
                batch_backward_data.append((sent, extract_triplets, one_backward, one_sent_score))
                batch_sents.append(sent)

            print("\nOptimize ...")
            oie_prompt_opt = kg4.optimizer_batch(batch_backward_data)

            if "<IMPROVED_PROMPT>" in oie_prompt_opt:
                oie_prompt_opt = (
                    oie_prompt_opt
                    .replace("<IMPROVED_PROMPT>", "")
                    .replace("</IMPROVED_PROMPT>", "")
                    .strip()
                )

            print("Optimize Eval ...")
            bat_score_opt, bat_eval_pairs_opt, bat_extract_triplets_opt = kg4.batch_extract_and_nli_eval(
                batch_sents, oie_prompt_opt
            )

            old_total = sum(batch_score)
            new_total = sum(bat_score_opt)

            print(f"OLD TOTAL SCORE: {old_total}")
            print(f"NEW TOTAL SCORE: {new_total}")

            if new_total > old_total:
                oie_prompt = oie_prompt_opt
                print("\n### PROMPT UPDATED ###")
                print(oie_prompt)
                print("OPTIMIZED TRIPLETS:")
                print(bat_extract_triplets_opt)
                print("OPTIMIZED SCORES:")
                print(bat_score_opt)

                batch_eval_pairs, batch_extract_triplets = bat_eval_pairs_opt, bat_extract_triplets_opt
            else:
                print("\nPrompt not updated.")

            print("Rel normalization & Write...")
            for _eval_pairs, _extract_triplets, _sent in zip(batch_eval_pairs, batch_extract_triplets, batch_sents):
                max_workers = max(1, min(os.cpu_count() or 4, len(_eval_pairs)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(lambda p: post_processor.process_pair(p, _sent), _eval_pairs))

                final_triplets = []
                entailment_triplets = []

                for result in results:
                    if result:
                        final_tri, entailment = result
                        final_triplets.append(final_tri)
                        if entailment == "entailment":
                            entailment_triplets.append(final_tri)

                out_raw.write(json.dumps(_extract_triplets, ensure_ascii=False) + "\n")
                out_raw.flush()

                out_en.write(json.dumps(entailment_triplets, ensure_ascii=False) + "\n")
                out_en.flush()

                out_f.write(json.dumps(final_triplets, ensure_ascii=False) + "\n")
                out_f.flush()


if __name__ == '__main__':
    save_paths = {
        "raw_triplets_path": f"./{save_dir}/{chose_dataset}/raw_triplets.txt",
        "entailment_tris_path": f"./{save_dir}/{chose_dataset}/only_entailment.txt",
        "final_triplets_path": f"./{save_dir}/{chose_dataset}/final_triplets.txt",
    }

    for path_key, path_value in save_paths.items():
        dir_path = os.path.dirname(path_value)
        os.makedirs(dir_path, exist_ok=True)

    main(chose_dataset, choose_llm, save_paths)