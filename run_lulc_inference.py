import os
import re
import ast
import json
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI


# CONFIG 
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

DATASET_PATH = "datasets/lulc_test.txt"
OUTPUT_DIR = "outputs/lulc_test"

RAW_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "raw_triplets.txt")
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "final_triplets.txt")
ERRORS_PATH = os.path.join(OUTPUT_DIR, "parsing_errors.txt")
FEW_SHOT_PATH = "./prompts/few_shot_examples/lulc_test/oie_few_shot_examples.txt"

# FINAL PROMPT
OIE_PROMPT = """Your task is to extract a semantic graph from the input text as a list of triples.

Return ONLY valid JSON in the following format:
[[subject, predicate, object], ...]
Do not include any explanation or additional text.

Extract only meaningful, non-redundant, and informative triples from the sentences. Focus on land use, land cover, land cover change, their drivers, impacts, and closely related environmental or socio-economic processes. Use only information explicitly stated in the text and DO NOT infer relations unless clearly expressed. If no clear and informative triples can be extracted, return an empty list.

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

When extracting triples, ensure that:
1. Subjects and objects are precise, identifiable, and unambiguous; subjects must precede objects in relational statements.
2. Analyze verb phrases and context carefully to align with the text. Only assign relationships if the text provides explicit support.
3. Conduct thorough examinations of all potential relationships, including implied interactions in subordinate clauses, to ensure comprehensive extraction of relevant interactions.
4. Prioritize clarity in argument direction; each triple must accurately reflect logical relations in accordance with the fixed canonical relation schema and maintain coherent entity boundaries.
5. Clearly define each entity and separate temporal expressions or geographic context from the main relational structure to enhance understanding.

Each extracted triple should represent distinct events or characteristics and strictly adhere to the fixed canonical relation schema, ensuring that all relevant relationships in the text are captured without ambiguity.
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

def get_client() -> OpenAI:
    if os.getenv("USE_OLLAMA") == "1":
        return OpenAI(
            api_key="ollama",
            base_url="http://localhost:11434/v1"
        )

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
                temperature=0,
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