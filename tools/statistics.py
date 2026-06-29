import pandas as pd

INPUT_FILE = "agreement.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_FILE = "per_sentence_metrics.xlsx"

MODELS = {
    "GPT4omini": ("GPT4ominiresults", "Status"),
    "LLaMa3": ("LLaMa3results", "Status.1"),
    "GPT55": ("GPT55results", "Status.2"),
    "DeepSeekR1": ("DeepSeekR1results", "Status.3"),
}

VALID = "VALID"
VALID_UNINF = "VALID_UNINF"


def is_extracted_triple(x):
    if pd.isna(x):
        return False

    x = str(x).strip()

    return x not in ["", "[]"]


df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

df["ID"] = df["ID"].ffill().astype(int)
df["sentence"] = df["sentence"].ffill()

rows = []

for sentence_id, group in df.groupby("ID"):

    row = {
        "ID": sentence_id,
        "Sentence": group["sentence"].iloc[0]
    }

    for model_name, (triple_col, status_col) in MODELS.items():

        extracted_mask = group[triple_col].apply(is_extracted_triple)

        statuses = (
            group[status_col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        extracted = extracted_mask.sum()

        valid = (
            (statuses == VALID) &
            extracted_mask
        ).sum()

        valid_plus_uninf = (
            ((statuses == VALID) | (statuses == VALID_UNINF))
            & extracted_mask
        ).sum()

        row[f"{model_name}_Extracted"] = extracted
        row[f"{model_name}_VALID"] = valid
        row[f"{model_name}_VALID+UNINF"] = valid_plus_uninf

    rows.append(row)

out_df = pd.DataFrame(rows)

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    out_df.to_excel(writer, index=False)

print(out_df)

print(f"\nSaved to: {OUTPUT_FILE}")