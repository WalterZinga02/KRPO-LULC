from pathlib import Path
import ast
import pandas as pd

# === FILE PATHS ===
SENTENCES_FILE = "sentences.txt"
TRIPLETS_FILE = "triplets.txt"
OUTPUT_FILE = "triplets_expanded.xlsx"

# === READ FILES ===
sentences = Path(SENTENCES_FILE).read_text(encoding="utf-8").splitlines()
triplets_lines = Path(TRIPLETS_FILE).read_text(encoding="utf-8").splitlines()

# === CHECK ===
if len(sentences) != len(triplets_lines):
    raise ValueError(
        f"Mismatch: {len(sentences)} sentences vs {len(triplets_lines)} triplet rows"
    )

rows = []

# === PROCESS ===
for idx, (sentence, triplet_line) in enumerate(zip(sentences, triplets_lines), start=1):

    try:
        triplets = ast.literal_eval(triplet_line)
    except Exception:
        triplets = []

    # Caso empty set
    if not triplets:
        rows.append({
            "id": idx,
            "sentence": sentence,
            "triplet": "[]"
        })

    # Una riga per tripla
    else:
        for triple in triplets:
            rows.append({
                "id": idx,
                "sentence": sentence,
                "triplet": str(triple)
            })

# === SAVE ===
df = pd.DataFrame(rows)

df.to_excel(OUTPUT_FILE, index=False)

print(f"Saved to: {OUTPUT_FILE}")