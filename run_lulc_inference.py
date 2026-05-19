import os
import re
import ast
import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
from openai import OpenAI

from post_processing import TripletPostProcessor


# CONFIG
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

DATASET_NAME = "lulc_sample"

DATASET_PATH = f"datasets/{DATASET_NAME}.txt"
OUTPUT_DIR = f"outputs/{DATASET_NAME}"

RAW_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "raw_triplets.txt")
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_triplets.txt")
ERRORS_PATH = os.path.join(OUTPUT_DIR, "parsing_errors.txt")

BENCHMARK_PATH = os.path.join(OUTPUT_DIR, "benchmark_results.jsonl")

FEW_SHOT_PATH = f"./prompts/few_shot_examples/{DATASET_NAME}/oie_few_shot_examples.txt"
SCHEMA_PATH = f"./schemas/{DATASET_NAME}_schema.csv"

# Tracking mode:
# - auto: EcoLogits for API/OpenAI, CodeCarbon for local/Ollama
# - api: force EcoLogits
# - local: force CodeCarbon
# - none: disable energy tracking
TRACKING_MODE = os.getenv("TRACKING_MODE", "auto").lower()


# SYSTEM PROMPT
OIE_PROMPT = """Our task is to extract a semantic graph from the input text as a list of triples.

Return ONLY valid JSON in the following format:
[["subject", "predicate", "object"], ...]
Do not include any explanation or additional text.

Extract only meaningful, non-redundant, and informative triples from the sentences. 
Focus on land use, land cover, land cover change, their drivers, impacts, and closely related environmental or socio-economic processes. 
Use only information explicitly stated in the text and DO NOT infer relations unless clearly expressed. 
If no clear and informative triples can be extracted, return an empty list.


Prioritize the following relations when they fit naturally:
- CAUSES: Direct causal relationship.
- CONVERTED_TO: Transformation from one state to another.
- LOCATED_IN: Spatial relationship.
- OCCURS_DURING: Temporal relationship.
- INCREASES: Growth or rise in quantity or quality.
- DECREASES: Reduction in quantity or quality.
- DOMINATES: Prevailing presence or influence.
- AFFECTS: Directional impact but not strictly causal.
- ASSOCIATED_WITH: Correlation without clear direction.

When extracting triples:
1. Subjects and objects must be precise.
2. Only use explicitly supported relations.
3. Avoid redundant or inferred triples.
4. Maintain correct directionality.

Each extracted triple should represent distinct events or characteristics and strictly adhere to the fixed
canonical relation schema, ensuring that all relevant relationships in the text are captured without ambiguity.
"""


# USER PROMPT
SUFFIX_PROMPT = """
Here are some examples:
{few_shot_examples}

Now please extract triplets from the following text.

Text:
{input_text}

Triplets:
"""


def validate_input_paths() -> None:
    required_files = [
        DATASET_PATH,
        FEW_SHOT_PATH,
        SCHEMA_PATH
    ]

    for path in required_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )


def read_schema_relations(path: str):
    local_rels = pd.read_csv(
        path,
        header=None,
        names=["name", "schema"]
    )

    return local_rels["name"].to_numpy()


def get_client() -> OpenAI:
    if os.getenv("USE_OLLAMA") == "1":
        return OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            timeout=60.0
        )

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set."
        )

    return OpenAI(
        api_key=api_key,
        timeout=60.0
    )


def is_local_run(model_name: str) -> bool:
    if os.getenv("USE_OLLAMA") == "1":
        return True

    model = model_name.lower()

    local_keywords = [
        "llama",
        "llama3",
        "deepseek",
        "mistral",
        "qwen",
        "gemma",
        "ollama"
    ]

    return any(keyword in model for keyword in local_keywords)


def get_tracking_backend(model_name: str) -> str:
    if TRACKING_MODE in {"api", "local", "none"}:
        return TRACKING_MODE

    if is_local_run(model_name):
        return "local"

    return "api"


def init_ecologits_if_needed(tracking_backend: str) -> None:
    if tracking_backend != "api":
        return

    try:
        from ecologits import EcoLogits
    except ImportError as exc:
        raise ImportError(
            "EcoLogits is required for API energy tracking. "
            "Install it with: pip install ecologits[openai]"
        ) from exc

    try:
        EcoLogits.init(providers=["openai"])
    except TypeError:
        EcoLogits.init(providers="openai")


def start_codecarbon_if_needed(
    tracking_backend: str,
    output_dir: str,
    model_name: str
):
    if tracking_backend != "local":
        return None

    try:
        from codecarbon import EmissionsTracker
    except ImportError as exc:
        raise ImportError(
            "CodeCarbon is required for local energy tracking. "
            "Install it with: pip install codecarbon"
        ) from exc

    codecarbon_dir = os.path.join(
        output_dir,
        "codecarbon"
    )

    Path(codecarbon_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    tracker = EmissionsTracker(
        project_name=f"{DATASET_NAME}_{sanitize_name(model_name)}",
        output_dir=codecarbon_dir,
        output_file="emissions.csv",
        save_to_file=True,
        log_level="error"
    )

    tracker.start()

    return tracker


def stop_codecarbon_tracker(tracker) -> Dict[str, Optional[float]]:
    if tracker is None:
        return {
            "energy_kwh": None,
            "co2_kg": None
        }

    emissions_kg = tracker.stop()

    energy_kwh = None

    final_data = getattr(tracker, "final_emissions_data", None)

    if final_data is not None:
        energy_kwh = getattr(final_data, "energy_consumed", None)

        if emissions_kg is None:
            emissions_kg = getattr(final_data, "emissions", None)

    return {
        "energy_kwh": energy_kwh,
        "co2_kg": emissions_kg
    }


def sanitize_name(name: str) -> str:
    return re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "_",
        name
    )


def read_sentences(path: str) -> List[str]:
    sentences: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            sentence = line.strip()

            if sentence:
                sentences.append(sentence)

    return sentences


def read_few_shot(path: Optional[str]) -> str:
    if not path:
        return ""

    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def extract_usage(completion) -> Dict[str, Optional[int]]:
    usage = getattr(completion, "usage", None)

    if usage is None:
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        }

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None)
    }


def extract_ecologits_impacts(completion) -> Dict[str, Optional[float]]:
    impacts = getattr(completion, "impacts", None)

    if impacts is None:
        return {
            "energy_kwh": None,
            "co2_kg": None
        }

    energy_kwh = None
    co2_kg = None

    try:
        energy_kwh = impacts.energy.value.mean
    except Exception:
        pass

    try:
        co2_kg = impacts.gwp.value.mean
    except Exception:
        pass

    return {
        "energy_kwh": energy_kwh,
        "co2_kg": co2_kg
    }


def add_optional_number(
    current: Optional[float],
    value: Optional[float]
) -> Optional[float]:
    if value is None:
        return current

    if current is None:
        return float(value)

    return current + float(value)


def call_llm(
    client: OpenAI,
    sentence: str,
    few_shot_examples: str,
    model_name: str,
    tracking_backend: str,
    max_retries: int = 5
) -> Tuple[Optional[str], Dict[str, Any]]:

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": OIE_PROMPT
                    },
                    {
                        "role": "user",
                        "content": SUFFIX_PROMPT.format(
                            few_shot_examples=few_shot_examples,
                            input_text=sentence
                        )
                    }
                ],
                max_completion_tokens=1200
            )

            content = completion.choices[0].message.content

            metrics = {
                **extract_usage(completion),
                "energy_kwh": None,
                "co2_kg": None
            }

            if tracking_backend == "api":
                metrics.update(
                    extract_ecologits_impacts(completion)
                )

            if isinstance(content, str) and content.strip():
                return content.strip(), metrics

        except Exception as e:
            wait_s = min(2 ** attempt, 60)

            print(
                f"[WARN] attempt "
                f"{attempt + 1}/{max_retries} failed: {e}"
            )

            time.sleep(wait_s)

    return None, {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "energy_kwh": None,
        "co2_kg": None
    }


def extract_list_block(text: str) -> str:
    text = text.strip()

    text = re.sub(
        r"^```(?:json|python)?\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    if text.startswith("[") and text.endswith("]"):
        return text

    match = re.search(
        r"\[\s*\[.*\]\s*\]",
        text,
        re.DOTALL
    )

    if not match:
        raise ValueError(
            "No valid list block found in model response."
        )

    return match.group(0)


def parse_triplets(text: str) -> List[List[str]]:
    list_block = extract_list_block(text)

    try:
        parsed = json.loads(list_block)

    except json.JSONDecodeError:
        parsed = ast.literal_eval(list_block)

    if not isinstance(parsed, list):
        raise ValueError("Parsed output is not a list.")

    clean_triplets: List[List[str]] = []

    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(
                f"Invalid triplet format: {item}"
            )

        head, relation, tail = item

        if not all(
            isinstance(x, str)
            for x in (head, relation, tail)
        ):
            raise ValueError(
                f"Triplet elements must be strings: {item}"
            )

        head = head.strip()
        relation = relation.strip()
        tail = tail.strip()

        if not head or not relation or not tail:
            continue

        clean_triplets.append([
            head,
            relation,
            tail
        ])

    return clean_triplets


def post_process_triplets(
    processor: TripletPostProcessor,
    triplets: List[List[str]],
    sentence: str
) -> List[List[str]]:

    final_triplets: List[List[str]] = []

    for triplet in triplets:
        pair = (
            triplet,
            "",
            "entailment"
        )

        result = processor.process_pair(
            pair,
            sentence
        )

        if result is None:
            continue

        clean_triplet, _entailment = result

        final_triplets.append(clean_triplet)

    deduped: List[List[str]] = []
    seen = set()

    for triplet in final_triplets:
        key = tuple(
            part.lower().strip()
            for part in triplet
        )

        if key not in seen:
            seen.add(key)
            deduped.append(triplet)

    return deduped


def ensure_output_dir(path: str) -> None:
    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


def append_triplets_txt(
    path: str,
    triplets: List[List[str]]
) -> None:

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                triplets,
                ensure_ascii=False
            ) + "\n"
        )


def append_error_txt(
    path: str,
    record: dict
) -> None:

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


def append_benchmark_jsonl(
    path: str,
    record: dict
) -> None:

    with open(path, "a", encoding="utf-8") as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                indent=2
            ) + "\n"
        )


def safe_divide(
    numerator: Optional[float],
    denominator: Optional[float]
) -> Optional[float]:

    if numerator is None:
        return None

    if denominator is None or denominator == 0:
        return None

    return numerator / denominator


def main() -> None:
    validate_input_paths()

    ensure_output_dir(OUTPUT_DIR)

    for path in [
        RAW_OUTPUT_PATH,
        FINAL_OUTPUT_PATH,
        ERRORS_PATH
    ]:
        with open(path, "w", encoding="utf-8"):
            pass

    logging.basicConfig(
        filename="postprocessing.log",
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    tracking_backend = get_tracking_backend(MODEL_NAME)

    init_ecologits_if_needed(tracking_backend)

    codecarbon_tracker = start_codecarbon_if_needed(
        tracking_backend=tracking_backend,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME
    )

    schema_relations = read_schema_relations(
        SCHEMA_PATH
    )

    processor = TripletPostProcessor(
        schema_relations,
        logger
    )

    client = get_client()

    few_shot_examples = read_few_shot(
        FEW_SHOT_PATH
    )

    sentences = read_sentences(
        DATASET_PATH
    )

    total = len(sentences)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    api_energy_kwh: Optional[float] = None
    api_co2_kg: Optional[float] = None

    total_raw_triplets = 0
    total_final_triplets = 0
    successful_responses = 0
    failed_responses = 0
    parsing_errors = 0

    script_start = time.time()

    print(f"Loaded {total} sentences from {DATASET_PATH}")
    print(f"Loaded schema from {SCHEMA_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Tracking backend: {tracking_backend}")

    try:
        for i, sentence in enumerate(sentences, start=1):
            print(f"[{i}/{total}] Processing...")

            response, llm_metrics = call_llm(
                client=client,
                sentence=sentence,
                few_shot_examples=few_shot_examples,
                model_name=MODEL_NAME,
                tracking_backend=tracking_backend
            )

            prompt_tokens = llm_metrics.get("prompt_tokens")
            completion_tokens = llm_metrics.get("completion_tokens")
            tokens = llm_metrics.get("total_tokens")

            if prompt_tokens is not None:
                total_prompt_tokens += int(prompt_tokens)

            if completion_tokens is not None:
                total_completion_tokens += int(completion_tokens)

            if tokens is not None:
                total_tokens += int(tokens)

            if tracking_backend == "api":
                api_energy_kwh = add_optional_number(
                    api_energy_kwh,
                    llm_metrics.get("energy_kwh")
                )

                api_co2_kg = add_optional_number(
                    api_co2_kg,
                    llm_metrics.get("co2_kg")
                )

            if response is None:
                failed_responses += 1

                append_triplets_txt(
                    RAW_OUTPUT_PATH,
                    []
                )

                append_triplets_txt(
                    FINAL_OUTPUT_PATH,
                    []
                )

                append_error_txt(
                    ERRORS_PATH,
                    {
                        "id": i - 1,
                        "sentence": sentence,
                        "error": "No response from model"
                    }
                )

                continue

            successful_responses += 1

            try:
                raw_triplets = parse_triplets(response)

                final_triplets = post_process_triplets(
                    processor=processor,
                    triplets=raw_triplets,
                    sentence=sentence
                )

                total_raw_triplets += len(raw_triplets)
                total_final_triplets += len(final_triplets)

                append_triplets_txt(
                    RAW_OUTPUT_PATH,
                    raw_triplets
                )

                append_triplets_txt(
                    FINAL_OUTPUT_PATH,
                    final_triplets
                )

            except Exception as e:
                parsing_errors += 1

                with open(
                    RAW_OUTPUT_PATH,
                    "a",
                    encoding="utf-8"
                ) as f:
                    f.write(response.strip() + "\n")

                append_triplets_txt(
                    FINAL_OUTPUT_PATH,
                    []
                )

                append_error_txt(
                    ERRORS_PATH,
                    {
                        "id": i - 1,
                        "sentence": sentence,
                        "response": response,
                        "error": str(e)
                    }
                )

    finally:
        total_runtime_sec = time.time() - script_start

        local_energy_data = stop_codecarbon_tracker(
            codecarbon_tracker
        )

        if tracking_backend == "local":
            total_energy_kwh = local_energy_data["energy_kwh"]
            total_co2_kg = local_energy_data["co2_kg"]
            energy_measurement_type = "measured_local_hardware_codecarbon"

        elif tracking_backend == "api":
            total_energy_kwh = api_energy_kwh
            total_co2_kg = api_co2_kg
            energy_measurement_type = "estimated_api_ecologits"

        else:
            total_energy_kwh = None
            total_co2_kg = None
            energy_measurement_type = "disabled"

        benchmark_record = {
            "model": MODEL_NAME,
            "dataset": DATASET_NAME,
            "sentences_total": total,
            "successful_responses": successful_responses,
            "failed_responses": failed_responses,
            "parsing_errors": parsing_errors,
            "total_runtime_sec": total_runtime_sec,
            "avg_runtime_sec_per_sentence": safe_divide(
                total_runtime_sec,
                total
            ),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_tokens_per_sentence": safe_divide(
                total_tokens,
                total
            ),
            "total_energy_kwh": total_energy_kwh,
            "total_co2_kg": total_co2_kg,
            "energy_kwh_per_sentence": safe_divide(
                total_energy_kwh,
                total
            ),
            "co2_kg_per_sentence": safe_divide(
                total_co2_kg,
                total
            ),
            "total_raw_triplets": total_raw_triplets,
            "total_final_triplets": total_final_triplets,
            "energy_kwh_per_final_triplet": safe_divide(
                total_energy_kwh,
                total_final_triplets
            ),
            "co2_kg_per_final_triplet": safe_divide(
                total_co2_kg,
                total_final_triplets
            ),
            "runtime_sec_per_final_triplet": safe_divide(
                total_runtime_sec,
                total_final_triplets
            ),
        }

        append_benchmark_jsonl(
            BENCHMARK_PATH,
            benchmark_record
        )

        print("Done.")
        print(f"Raw triplets:       {RAW_OUTPUT_PATH}")
        print(f"Final triplets:     {FINAL_OUTPUT_PATH}")
        print(f"Errors:             {ERRORS_PATH}")
        print(f"Benchmark metrics:  {BENCHMARK_PATH}")
        print(
            "Benchmark summary:\n"
            + json.dumps(
                benchmark_record,
                indent=2,
                ensure_ascii=False
            )
        )


if __name__ == "__main__":
    main()