import json
from pathlib import Path

import pandas as pd


BENCHMARK_FILES = [
    "benchmarkGPT4omini.jsonl",
    "benchmarkGPT55.jsonl",
    "benchmarkLLaMa3.jsonl",
    "benchmarkDeepSeekR1.jsonl",
]

OUTPUT_FILE = "benchmark_comparison.xlsx"

API_PRICES_PER_1M = {
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-5.5": {
        "input": 5.00,
        "output": 30.00,
    },
}


def safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def load_json_or_jsonl(path: Path):
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"{path} is empty")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        return lines[-1]


rows = []

for file in BENCHMARK_FILES:
    path = Path(file)

    if not path.exists():
        print(f"[WARN] File not found, skipped: {file}")
        continue

    data = load_json_or_jsonl(path)

    model = data.get("model", path.stem)

    prompt_tokens = data.get("total_prompt_tokens", 0) or 0
    completion_tokens = data.get("total_completion_tokens", 0) or 0
    total_tokens = data.get("total_tokens", prompt_tokens + completion_tokens) or 0

    sentences = data.get("sentences_total", 0) or 0
    successful = data.get("successful_responses")
    failed = data.get("failed_responses")
    parsing_errors = data.get("parsing_errors")

    runtime_sec = data.get("total_runtime_sec")
    energy_kwh = data.get("total_energy_kwh")
    co2_kg = data.get("total_co2_kg")

    raw_triplets = data.get("total_raw_triplets", 0) or 0
    final_triplets = data.get("total_final_triplets", 0) or 0

    input_price = API_PRICES_PER_1M.get(model, {}).get("input")
    output_price = API_PRICES_PER_1M.get(model, {}).get("output")

    api_cost_usd = None
    if input_price is not None and output_price is not None:
        api_cost_usd = (
            prompt_tokens / 1_000_000 * input_price
            + completion_tokens / 1_000_000 * output_price
        )

    rows.append({
        "model": model,
        "type": "API" if model in API_PRICES_PER_1M else "local",

        # Total / overview metrics
        "sentences": sentences,
        "successful": successful,
        "failed": failed,
        "parsing_errors": parsing_errors,

        "runtime_sec": runtime_sec,

        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,

        "energy_kwh": energy_kwh,
        "co2_kg": co2_kg,

        "raw_triplets": raw_triplets,
        "final_triplets": final_triplets,

        "api_input_price_per_1M": input_price,
        "api_output_price_per_1M": output_price,
        "api_cost_usd": api_cost_usd,

        # Normalized per sentence
        "runtime_sec_per_sentence": safe_div(runtime_sec, sentences),
        "prompt_tokens_per_sentence": safe_div(prompt_tokens, sentences),
        "completion_tokens_per_sentence": safe_div(completion_tokens, sentences),
        "total_tokens_per_sentence": safe_div(total_tokens, sentences),
        "energy_kwh_per_sentence": safe_div(energy_kwh, sentences),
        "co2_kg_per_sentence": safe_div(co2_kg, sentences),
        "api_cost_usd_per_sentence": safe_div(api_cost_usd, sentences),
        "raw_triplets_per_sentence": safe_div(raw_triplets, sentences),
        "final_triplets_per_sentence": safe_div(final_triplets, sentences),

        # Normalized per final triplet
        "runtime_sec_per_final_triplet": safe_div(runtime_sec, final_triplets),
        "prompt_tokens_per_final_triplet": safe_div(prompt_tokens, final_triplets),
        "completion_tokens_per_final_triplet": safe_div(completion_tokens, final_triplets),
        "total_tokens_per_final_triplet": safe_div(total_tokens, final_triplets),
        "energy_kwh_per_final_triplet": safe_div(energy_kwh, final_triplets),
        "co2_kg_per_final_triplet": safe_div(co2_kg, final_triplets),
        "api_cost_usd_per_final_triplet": safe_div(api_cost_usd, final_triplets),

        # Efficiency ratios
        "final_triplets_per_kwh": safe_div(final_triplets, energy_kwh),
        "final_triplets_per_kg_co2": safe_div(final_triplets, co2_kg),
    })


df = pd.DataFrame(rows)

overview_cols = [
    "model",
    "type",
    "sentences",
    "successful",
    "failed",
    "parsing_errors",
    "runtime_sec",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "energy_kwh",
    "co2_kg",
    "raw_triplets",
    "final_triplets",
    "api_input_price_per_1M",
    "api_output_price_per_1M",
    "api_cost_usd",
]

normalized_per_sentence_cols = [
    "model",
    "type",
    "runtime_sec_per_sentence",
    "prompt_tokens_per_sentence",
    "completion_tokens_per_sentence",
    "total_tokens_per_sentence",
    "energy_kwh_per_sentence",
    "co2_kg_per_sentence",
    "api_cost_usd_per_sentence",
    "raw_triplets_per_sentence",
    "final_triplets_per_sentence",
]

normalized_per_triplet_cols = [
    "model",
    "type",
    "runtime_sec_per_final_triplet",
    "prompt_tokens_per_final_triplet",
    "completion_tokens_per_final_triplet",
    "total_tokens_per_final_triplet",
    "energy_kwh_per_final_triplet",
    "co2_kg_per_final_triplet",
    "api_cost_usd_per_final_triplet",
    "final_triplets_per_kwh",
    "final_triplets_per_kg_co2",
]

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df[overview_cols].to_excel(
        writer,
        sheet_name="overview",
        index=False,
    )

    df[normalized_per_sentence_cols].to_excel(
        writer,
        sheet_name="normalized_per_sentence",
        index=False,
    )

    df[normalized_per_triplet_cols].to_excel(
        writer,
        sheet_name="normalized_per_triplet",
        index=False,
    )

print(f"Created {OUTPUT_FILE}")