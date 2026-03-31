import os
import re
import ast
import json
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI


# CONFIG 
MODEL_NAME = "gpt-4o-mini"

DATASET_PATH = "datasets/lulc_dataset.txt"
OUTPUT_DIR = "outputs/lulc_full_inference"

RAW_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "raw_triplets.txt")
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_triplets.txt")
ERRORS_PATH = os.path.join(OUTPUT_DIR, "parsing_errors.txt")
FEW_SHOT_PATH = "./prompts/few_shot_examples/lulc_dataset/oie_few_shot_examples.txt"

# FINAL PROMPT
OIE_PROMPT = """Your task is to transform the given text into a semantic graph in the form of a comprehensive list of triplets. The triplets must be in the format [Entity1, Relationship, Entity2]. The domain is land use and land cover.

1. Ensure that each extracted triplet is directly supported by explicit statements in the text and reflects clear, unambiguous relationships. Validate relationships against the context for logical consistency.
2. Identify and extract all relevant entities, including interactions between different land cover types, environmental factors, and demographic influences. Consider broader implications of land use changes.
3. Use precise and descriptive relationship labels. For example, use terms like "INCREASES_BY," "DECREASES_BY," "IS_A_TYPE_OF," "AFFECTS," and "CAUSES." Avoid vague terms.
4. Include relevant time frames or conditions to clarify the context of changes, ensuring that temporal references are explicitly linked to the entities involved.
5. Avoid extracting uncertain relationships; only include those that are well-supported by the context. Provide context-specific qualifiers for entities and relationships to enhance clarity.

In your answer, please strictly only include the triplets list and do not include any explanation or apologies.
"""

SUFFIX_PROMPT = """
Here are some examples:
{few_shot_examples}

Now please extract triplets from the following text.
Text: {input_text}
Triplets: """

# Check that required input files (dataset and few-shot examples) exist before execution
def validate_input_paths() -> None:
    required_files = [DATASET_PATH, FEW_SHOT_PATH]
    for path in required_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required input file not found: {path}")

# Initialize OpenAI client using API key from environment variables
def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)

# Read dataset file and return a list of non-empty sentences (one per line)
def read_sentences(path: str) -> List[str]:
    sentences: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                sentences.append(s)
    return sentences

# Load few-shot examples from file as a single string (used in prompt construction)
def read_few_shot(path: Optional[str]) -> str:
    if not path:
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# Build full prompt by combining base instructions, few-shot examples, and input sentence
def build_prompt(sentence: str, few_shot_examples: str) -> str:
    return OIE_PROMPT + SUFFIX_PROMPT.format(
        few_shot_examples=few_shot_examples,
        input_text=sentence
    )

# Send prompt to LLM with retry logic and exponential backoff, return response text
def call_llm(client: OpenAI, prompt: str, model_name: str, max_retries: int = 5) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            content = completion.choices[0].message.content
            if isinstance(content, str) and content.strip():
                return content
        except Exception as e:
            wait_s = 2 ** attempt
            print(f"[WARN] attempt {attempt + 1}/{max_retries} failed: {e}")
            time.sleep(wait_s)
    return None

# Extract the list-like structure (triplets) from model response using regex if needed
def extract_list_block(text: str) -> str:
    text = text.strip()

    if text.startswith("[") and text.endswith("]"):
        return text

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("No valid list block found in model response.")
    return match.group(0)

# Parse and validate triplets from model response into a clean list of [head, relation, tail]
def parse_triplets(text: str) -> List[List[str]]:
    list_block = extract_list_block(text)
    parsed = ast.literal_eval(list_block)

    if not isinstance(parsed, list):
        raise ValueError("Parsed output is not a list.")

    clean_triplets: List[List[str]] = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"Invalid triplet format: {item}")
        h, r, t = item
        if not all(isinstance(x, str) for x in (h, r, t)):
            raise ValueError(f"Triplet elements must be strings: {item}")
        clean_triplets.append([h.strip(), r.strip(), t.strip()])

    return clean_triplets

# Create output directory if it does not already exist
def ensure_output_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

# Append a list of triplets as a JSON-formatted line to a text file
def append_triplets_txt(path: str, triplets: List[List[str]]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(triplets, ensure_ascii=False) + "\n")

# Append error information to a log file
def append_error_txt(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# MAIN
def main() -> None:
    validate_input_paths()
    ensure_output_dir(OUTPUT_DIR)

    for p in [RAW_OUTPUT_PATH, FINAL_OUTPUT_PATH, ERRORS_PATH]:
        with open(p, "w", encoding="utf-8"):
            pass

    client = get_client()
    few_shot_examples = read_few_shot(FEW_SHOT_PATH)
    sentences = read_sentences(DATASET_PATH)

    total = len(sentences)
    print(f"Loaded {total} sentences from {DATASET_PATH}")

    for i, sentence in enumerate(sentences, start=1):
        print(f"[{i}/{total}] Processing...")

        prompt = build_prompt(sentence, few_shot_examples)
        response = call_llm(client, prompt, MODEL_NAME)

        if response is None:
            append_triplets_txt(RAW_OUTPUT_PATH, [])
            append_triplets_txt(FINAL_OUTPUT_PATH, [])
            append_error_txt(ERRORS_PATH, {
                "id": i - 1,
                "sentence": sentence,
                "error": "No response from model"
            })
            continue

        try:
            triplets = parse_triplets(response)
            append_triplets_txt(RAW_OUTPUT_PATH, triplets)
            append_triplets_txt(FINAL_OUTPUT_PATH, triplets)

        except Exception as e:
            append_triplets_txt(RAW_OUTPUT_PATH, [])
            append_triplets_txt(FINAL_OUTPUT_PATH, [])
            append_error_txt(ERRORS_PATH, {
                "id": i - 1,
                "sentence": sentence,
                "response": response,
                "error": str(e)
            })

    print("Done.")
    print(f"Raw triplets:   {RAW_OUTPUT_PATH}")
    print(f"Final triplets: {FINAL_OUTPUT_PATH}")
    print(f"Errors:         {ERRORS_PATH}")


if __name__ == "__main__":
    main()