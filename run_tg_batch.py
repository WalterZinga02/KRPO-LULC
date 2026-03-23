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

choose_llm = "gpt-4o-mini"
_datasets = ["example", "rebel", "webnlg", "wiki-nre"]
chose_dataset = "example"
save_dir = ""
log_rm = ""

os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), exist_ok=True)

main_logger = logging.getLogger('main')
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", f"{save_dir}{log_rm}{chose_dataset}_inout.log")
main_handler = UTF8FileHandler(log_file, encoding='utf-8')
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


oie_prompt = """Your task is to transform the given text into a semantic graph in the form of a comprehensive list of triplets.
The triplets must be in the form of [Entity1, Relationship, Entity2].
In your answer, please strictly only include the triplets list and do not include any explanation or apologies.
"""
suffix_prompt = """
Here are some examples:
{few_shot_examples}

Now please extract triplets from the following text.
Text: {input_text}
Triplets: """


class KG4:
    def __init__(self, this_dataset, this_llm):
        llm_loger = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./logs/{save_dir}{log_rm}{chose_dataset}_log.log")
        self.llm = LLMInvoker(this_llm, llm_loger)
        self.data_sentences = read_sentences_from_file(f"./datasets/{this_dataset}.txt")
        # --- prompt txt ---,
        self.extract_few_shot_examples = open(f"./prompts/few_shot_examples/{this_dataset}/oie_few_shot_examples.txt",
                                              encoding='utf-8').read()

        self.triplet2text_prompt_template = open("./prompts/main_prompt/triplet2text.txt", encoding='utf-8').read()
        self.nli_few_shot_examples = open("./prompts/few_shot_examples/NLI.txt", encoding='utf-8').read()
        self.nli_prompt_template = open("./prompts/main_prompt/NLI.txt", encoding='utf-8').read()

        self.backward_1_template = BACKWARD_prompt_EVAL
        self.backward_2_template = BACKWARD_prompt_PRED
        self.optimizer_examples = OPTIMIZER_EXAMPLE_TEMPLATE
        self.optimizer_template = OPTIMIZER_prompt

        self.rel_simi_choice_template = open("./prompts/main_prompt/rel_simi_choice.txt", encoding='utf-8').read()

        local_rels = pd.read_csv(
            f"./schemas/{this_dataset}_schema.csv",
            header=None,
            names=["name", "schema"]
        )
        self.rels = local_rels["name"].to_numpy()
        self.rel_schemas = local_rels["schema"].to_numpy()

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
                main_logger.error("❌ LLM  - None, max tries reached")
                return []

            try:
                extract_res = extract_res.strip()
                if not (extract_res.startswith('[') and extract_res.endswith(']')):
                    match = re.search(r"\[.*\]", extract_res, re.DOTALL)
                    if not match:
                        raise ValueError("No valid list found in response")
                    extract_res = match.group(0)
                extract_res_list = ast.literal_eval(extract_res)
            except (ValueError, SyntaxError) as e:
                main_logger.error(f"❌ extract res {repr(extract_res)} fail：{e}")
                main_logger.warning(f"⚠️ json error {retry_count + 1} try: {e}")
                time.sleep(1)
                continue

            try:
                for tri in extract_res_list:
                    if len(tri) != 3:
                        raise ValueError("extract res not triplet")
                    head, relation, tail = tri
                    if not all(isinstance(x, str) for x in (head, relation, tail)):
                        raise ValueError("triplets must be strings")
            except (TypeError, ValueError, KeyError) as e:
                main_logger.warning(
                    f"⚠️ The triple is in the wrong format and is in progress. {retry_count + 1} try: {e}")
                time.sleep(1)
                continue

            main_logger.info(
                f"### currentProcessingSentence: {repr(sentence)}\nextractTripletResults{extract_res_list}")
            return extract_res_list

        main_logger.error("❌ Maximum number of retries reached and no valid triplet is obtained")
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
            f"\n{'-' * 50}\nNLI sent:\n{raw_sentence=}\n{sent_by_triplet=}\nNLI Response:\n{nli_result}\n{'-' * 50}\n")

        try:
            json_match = re.search(r'```json\s*(.*?)```', nli_result, flags=re.DOTALL)
            if json_match:
                nli_result = json_match.group(1).strip()
            nli_result = re.sub(r',\s*"reasoning"\s*:\s*".*$', '}', nli_result, flags=re.DOTALL)
            nli_res = json.loads(nli_result.replace('\n', ''))
            return nli_res
        except (ValueError, SyntaxError) as e:
            main_logger.error(f"❌ Single sentence by triple NLI result {repr(nli_result)} parseFailed{e}")
            return self.nli_single(raw_sentence, sent_by_triplet)

    def get_simi_rel_by_relcanon(self, rel, relation_text, raw_sent, rel_tri,
                             threshold1: float = 0.5, threshold2: float = 0.85):
        if rel in self.rels.tolist():
            return rel
        self.add_rel_schema(rel, relation_text)
        return rel

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
        optimize_1_prompt = self.backward_1_template.replace("__predict_triplets__", str(res_extract)).replace(
            "__evaluate_result__", _eval_str)
        main_logger.info(f"\nbackward_1...\n{'-' * 50}\n")
        main_logger.info(f"\n{'-' * 50}\nBackward_1:\n{res_extract=}\n{_eval_str=}\n")
        optimize_1 = None
        while optimize_1 is None:
            optimize_1 = self.llm.llm_chat_response(optimize_1_prompt)
        main_logger.info(f"\nBackward1 Response:\n{optimize_1}\n{'-' * 50}\n")
        return optimize_1

    def backward_2(self, input_sent, res_extract, backward1):
        global oie_prompt
        optimize_2_prompt = self.backward_2_template.replace("__sys_prompt__", oie_prompt).replace(
            "__input_sentence__", input_sent).replace("__extract_response__", str(res_extract)).replace(
            "__backward_response__", backward1)
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
        _sample_example = self.optimizer_examples.replace("__sys_prompt__", oie_prompt).replace("__input_sentence__",
                                                                                                one_in).replace(
            "__extract_response__", str(one_out)).replace("__feedback__", one_bak)

        optimizer_info = self.optimizer_template.replace("__sys_prompt__", oie_prompt).replace(
            "__update_insert_examples__", _sample_example)
        main_logger.info(f"\nOptimizer update prompt...\n{'-' * 50}\n")
        res_optimize = self.llm.llm_chat_response(optimizer_info)
        return res_optimize

    def optimizer_batch(self, batch_backward):
        global oie_prompt
        infos = []
        for b_in, b_out, b_bak, _ in batch_backward:
            _sample_example = self.optimizer_examples.replace("__sys_prompt__", oie_prompt).replace(
                "__input_sentence__", b_in).replace("__extract_response__", str(b_out)).replace("__feedback__", b_bak)
            infos.append(_sample_example)
        info_placeholder = "\n\n".join(infos)
        optimizer_info = self.optimizer_template.replace("__sys_prompt__", oie_prompt).replace(
            "__update_insert_examples__", info_placeholder)
        res_optimize = self.llm.llm_chat_response(optimizer_info)
        return res_optimize



def process_pair(pair, kg4, sent):
    try:
        tri_, tri_text, entailment = pair
        _h, _r, _t = tri_
    except Exception:
        return None

    rel_def = replace_entities(tri_text, _h, _t)
    simi_rel = kg4.get_simi_rel_by_relcanon(_r, rel_def, sent, tri_)
    final_tri = [_h, simi_rel, _t]
    return final_tri, entailment


def batch_iter(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield [(i + j, sent) for j, sent in enumerate(data[i:i + batch_size])]


def main(dataset, llm_model, output_paths):
    kg4 = KG4(dataset, llm_model)
    global oie_prompt
    batch_size = 5
    with open(output_paths["raw_triplets_path"], 'a', encoding='utf-8') as out_raw, \
            open(output_paths["entailment_tris_path"], 'a', encoding='utf-8') as out_en, \
            open(output_paths["final_triplets_path"], 'a', encoding='utf-8') as out_f:
        for b_idx, batch in enumerate(tqdm(batch_iter(kg4.data_sentences, batch_size),
                                         total=(len(kg4.data_sentences) + batch_size - 1) // batch_size)):
            batch_backward_data = []  # loss
            batch_sents = []
            batch_score, batch_eval_pairs, batch_extract_triplets = [], [], []

            for sid, sent in batch:
                print(f"sid {sid}")
                print("Extract & NLI ...")
                one_sent_score, eval_pairs, extract_triplets = kg4.extract_and_nli_eval(sent, oie_prompt)
                
                #debug info
                print("\n=== INITIAL PROMPT ===")
                print(oie_prompt)
                print("\n=== INITIAL TRIPLETS ===")
                print(extract_triplets)
                print("\n=== INITIAL SCORE ===")
                print(one_sent_score)

                batch_score.append(one_sent_score)
                batch_eval_pairs.append(eval_pairs)
                batch_extract_triplets.append(extract_triplets)
                print("Backward ...")
                one_backward = kg4.backward(sent, extract_triplets, eval_pairs, one_sent_score)
                batch_backward_data.append((sent, extract_triplets, one_backward, one_sent_score))
                batch_sents.append(sent)
            print("Optimize ...")
            oie_prompt_opt = kg4.optimizer_batch(batch_backward_data)

            #debug info
            print("\n=== OPTIMIZED PROMPT ===")
            print(oie_prompt_opt)

            if "<IMPROVED_PROMPT>" in oie_prompt_opt:
                oie_prompt_opt = oie_prompt_opt.replace("<IMPROVED_PROMPT>", "").replace("</IMPROVED_PROMPT>", "").strip()
            print("Optimize Eval ...")
            bat_score_opt, bat_eval_pairs_opt, bat_extract_triplets_opt = kg4.batch_extract_and_nli_eval(batch_sents,
                                                                                                         oie_prompt_opt)
            
            #debug info
            print("\n=== OPTIMIZED TRIPLETS ===")
            print(bat_extract_triplets_opt)
            print("\n=== OPTIMIZED SCORES ===")
            print(bat_score_opt)
            print("\n=== OLD TOTAL SCORE ===", sum(batch_score))
            print("=== NEW TOTAL SCORE ===", sum(bat_score_opt))

            if sum(bat_score_opt) > sum(batch_score):
                oie_prompt = oie_prompt_opt
                print(f"# After sample {sid}, OIE Prompt updated:\n{oie_prompt_opt}")
                batch_eval_pairs, batch_extract_triplets = bat_eval_pairs_opt, bat_extract_triplets_opt

            print("Rel normalization & Write...")
            for _eval_pairs, _extract_triplets, _sent in zip(batch_eval_pairs, batch_extract_triplets, batch_sents):
                max_workers = max(1, min(os.cpu_count() or 4, len(_eval_pairs)))
                results = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(lambda p: process_pair(p, kg4, _sent), _eval_pairs))

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
