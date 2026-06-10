import ast
import pandas as pd
from pathlib import Path


# =========================
# CONFIG
# =========================

INPUT_FILE = "Agreement.xlsx"
OUTPUT_EXCEL = "valid_triples_stats.xlsx"

ID_COL = "ID"
SENTENCE_COL = "sentence"

MODELS = [
    {
        "name": "GPT4omini",
        "triple_col": "GPT4ominiresults",
        "status_col": "Status",
        "txt_file": "valid_GPT4omini.txt",
    },
    {
        "name": "LLaMa3",
        "triple_col": "LLaMa3results",
        "status_col": "Status.1",
        "txt_file": "valid_LLaMa3.txt",
    },
    {
        "name": "GPT55",
        "triple_col": "GPT55results",
        "status_col": "Status.2",
        "txt_file": "valid_GPT55.txt",
    },
    {
        "name": "DeepSeekR1",
        "triple_col": "DeepSeekR1results",
        "status_col": "Status.3",
        "txt_file": "valid_DeepSeekR1.txt",
    },
]


# =========================
# UTILS
# =========================

def normalize_status(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def parse_triple(value):
    """
    Converte una cella Excel in una tripla Python.
    Accetta formati tipo:
    ['a', 'REL', 'b']
    oppure testo semplice.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "" or value == "[]":
        return None

    try:
        parsed = ast.literal_eval(value)

        if isinstance(parsed, list) and len(parsed) == 3:
            return parsed

        if isinstance(parsed, tuple) and len(parsed) == 3:
            return list(parsed)

    except Exception:
        pass

    return None


def is_real_triple(value):
    return parse_triple(value) is not None


# =========================
# LOAD DATA
# =========================

df = pd.read_excel(INPUT_FILE)

df[ID_COL] = df[ID_COL].ffill()
df[SENTENCE_COL] = df[SENTENCE_COL].ffill()
df[ID_COL] = df[ID_COL].astype(int)


# =========================
# TXT VALID TRIPLES PER MODEL
# =========================

for model in MODELS:
    triple_col = model["triple_col"]
    status_col = model["status_col"]
    txt_file = model["txt_file"]

    lines = []

    for sentence_id, group in df.groupby(ID_COL, sort=False):
        valid_triples = []

        for _, row in group.iterrows():
            status = normalize_status(row[status_col])
            triple = parse_triple(row[triple_col])

            if status == "VALID" and triple is not None:
                valid_triples.append(triple)

        lines.append(str(valid_triples))

    Path(txt_file).write_text("\n".join(lines), encoding="utf-8")

    print(f"Creato: {txt_file}")


# =========================
# VALID COUNT
# =========================

valid_count_rows = []

for sentence_id, group in df.groupby(ID_COL, sort=False):
    sentence = group[SENTENCE_COL].iloc[0]

    row = {
        "id_frase": sentence_id,
        "frase": sentence,
    }

    for model in MODELS:
        model_name = model["name"]
        triple_col = model["triple_col"]
        status_col = model["status_col"]

        valid_count = 0

        for _, r in group.iterrows():
            if normalize_status(r[status_col]) == "VALID" and is_real_triple(r[triple_col]):
                valid_count += 1

        row[f"{model_name}_valid_count"] = valid_count

    valid_count_rows.append(row)

valid_count_df = pd.DataFrame(valid_count_rows)


# =========================
# VALID PERCENTAGE
# =========================

valid_percentage_rows = []

for sentence_id, group in df.groupby(ID_COL, sort=False):
    sentence = group[SENTENCE_COL].iloc[0]

    row = {
        "id_frase": sentence_id,
        "frase": sentence,
    }

    for model in MODELS:
        model_name = model["name"]
        triple_col = model["triple_col"]
        status_col = model["status_col"]

        total_triples = 0
        valid_triples = 0

        for _, r in group.iterrows():
            triple = parse_triple(r[triple_col])

            if triple is not None:
                total_triples += 1

                if normalize_status(r[status_col]) == "VALID":
                    valid_triples += 1

        if total_triples == 0:
            percentage = ""
        else:
            percentage = valid_triples / total_triples

        row[f"{model_name}_valid_percentage"] = percentage

    valid_percentage_rows.append(row)

valid_percentage_df = pd.DataFrame(valid_percentage_rows)


# =========================
# EXPORT EXCEL
# =========================

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    valid_count_df.to_excel(writer, sheet_name="Valid_count", index=False)
    valid_percentage_df.to_excel(writer, sheet_name="Valid_percentage", index=False)

print(f"Creato: {OUTPUT_EXCEL}")