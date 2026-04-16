import random
from pathlib import Path

# === Config ===
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "datasets" / "lulc_dataset.txt"
OUTPUT_FILE = BASE_DIR / "datasets" / "lulc_sample.txt"
N_SAMPLES = 300
SEED = 42

def main():
    random.seed(SEED)

    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    with input_path.open("r", encoding="utf-8") as f:
        sentences = [s.rstrip("\n") for s in f]

    sampled_sentences = random.sample(sentences, N_SAMPLES)

    with output_path.open("w", encoding="utf-8") as f:
        for sentence in sampled_sentences:
            f.write(sentence + "\n")

    print(f"Campione di {N_SAMPLES} frasi salvato in: {output_path}")

if __name__ == "__main__":
    main()