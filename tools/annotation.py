from pathlib import Path
import ast
import pandas as pd


SENTENCES_FILE = "sentences.txt"

TRIPLETS_FILES = [
    "GPT4ominiresults.txt",
    "LLaMa3results.txt",
    "GPT5miniresults.txt",
    "GPT55results.txt",
]

OUTPUT_FILE = "triplets_expanded.xlsx"


if not 1 <= len(TRIPLETS_FILES) <= 4:
    raise ValueError("You must provide between 1 and 4 triplet files.")


def parse_triplets(line: str):
    try:
        triplets = ast.literal_eval(line)
        if isinstance(triplets, list):
            return triplets
    except Exception:
        pass
    return []


sentences = Path(SENTENCES_FILE).read_text(encoding="utf-8").splitlines()

column_names = [Path(file).stem for file in TRIPLETS_FILES]

all_triplets = []

for file in TRIPLETS_FILES:
    lines = Path(file).read_text(encoding="utf-8").splitlines()

    if len(lines) != len(sentences):
        raise ValueError(
            f"Mismatch in {file}: {len(sentences)} sentences vs {len(lines)} triplet rows"
        )

    all_triplets.append([parse_triplets(line) for line in lines])


rows = []

for idx, sentence in enumerate(sentences, start=1):
    triplets_per_file = [triplet_file[idx - 1] for triplet_file in all_triplets]

    max_len = max((len(t) for t in triplets_per_file), default=0)

    if max_len == 0:
        row = {
            "id": idx,
            "sentence": sentence,
        }

        for column_name in column_names:
            row[column_name] = "[]"

        rows.append(row)

    else:
        for j in range(max_len):
            row = {
                "id": idx if j == 0 else "",
                "sentence": sentence if j == 0 else "",
            }

            for column_name, triplets in zip(column_names, triplets_per_file):
                row[column_name] = (
                    str(triplets[j]) if j < len(triplets) else ""
                )

            rows.append(row)


df = pd.DataFrame(rows)
df.to_excel(OUTPUT_FILE, index=False)

print(f"Saved to: {OUTPUT_FILE}")