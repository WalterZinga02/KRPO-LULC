from pathlib import Path
import ast
import pandas as pd

SENTENCES_FILE = "sentences.txt"
TRIPLETS_FILE = "triplets.txt"
OUTPUT_FILE = "triplets_expanded_compact.xlsx"

sentences = Path(SENTENCES_FILE).read_text(encoding="utf-8").splitlines()
triplets_lines = Path(TRIPLETS_FILE).read_text(encoding="utf-8").splitlines()

if len(sentences) != len(triplets_lines):
    raise ValueError(
        f"Mismatch: {len(sentences)} sentences vs {len(triplets_lines)} triplet rows"
    )

rows = []

for idx, (sentence, triplet_line) in enumerate(zip(sentences, triplets_lines), start=1):

    try:
        triplets = ast.literal_eval(triplet_line)
    except Exception:
        triplets = []

    if not triplets:
        rows.append({
            "id": idx,
            "sentence": sentence,
            "triplet": "[]"
        })
    else:
        for j, triple in enumerate(triplets):
            rows.append({
                "id": idx if j == 0 else "",
                "sentence": sentence if j == 0 else "",
                "triplet": str(triple)
            })

df = pd.DataFrame(rows)
df.to_excel(OUTPUT_FILE, index=False)

print(f"Saved to: {OUTPUT_FILE}")