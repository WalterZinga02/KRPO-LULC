import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs and known journal/footer noise
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(
        r"rstb\.royalsocietypublishing\.org\s+Phil\s+Trans\s+R\s+Soc\s+B\s+\d+:\s*\d+",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove raw TEI/XML metadata
    # Examples: @level, @type,  @target
    text = re.sub(r"@\w+\s*:\s*[A-Za-z0-9_#\-.]+", " ", text)
    text = re.sub(r"#\w+\s*:\s*[A-Za-z0-9_#\-.]+", " ", text)
    text = re.sub(r"'@type'\s*:\s*'[^']*'", " ", text)
    text = re.sub(r"'@target'\s*:\s*'[^']*'", " ", text)
    text = re.sub(r"'@xmlns'\s*:\s*'[^']*'", " ", text)
    text = re.sub(r"'#text'\s*:\s*'[^']*'", " ", text)

    # Remove section/header markers
    text = re.sub(
        r"\b(Title Content|Abstract Content|Body Content)\b\s*:?",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove isolated structural TEI tokens like "p", "div", "head", "ref"
    text = re.sub(r"\b(div|head|body|ref|p)\b", " ", text, flags=re.IGNORECASE)

    # Remove numeric citations like [1], [8, 9]
    text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", " ", text)

    # Keep annotated words but remove square brackets
    # Example: [deforestation] -> deforestation
    text = re.sub(r"\[([A-Za-z][^\[\]]*?)\]", r"\1", text)

    # Remove figure/table/footnote references
    text = re.sub(r"'(?:fig|tab)_\d+'", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:fig|tab)_\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(Table|Figure|Fig\.?)\s+[A-Za-z]?\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"#\w+(?:_\d+)?", " ", text)
    text = re.sub(r"\bfoot\s+foot_\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bfoot_\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b@place\s+foot\b", " ", text, flags=re.IGNORECASE)

    # Remove dictionary-like keys
    # Example: 'something':
    text = re.sub(r"'[^']*'\s*:\s*", " ", text)

    # Remove serialized-list junk
    text = re.sub(r"\bNone\b", " ", text)

    # Remove chunks like: ',,,,,,,, '
    text = re.sub(r"'\s*[.,;:]+\s*'", " ", text)

    # Remove quoted short garbage fragments
    #text = re.sub(r"'[^']{0,25}'", " ", text)
    text = re.sub(r"'(?:fig|tab)_\d+'", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"'\s*[.,;:]+\s*'", " ", text)

    # Remove large comma-separated numeric blocks
    text = re.sub(r"\b\d+(?:,\d+){2,}\b", " ", text)

    # Remove stray braces/brackets/quotes
    text = re.sub(r"[{}\[\]]", " ", text)
    text = re.sub(r"[\"`]+", " ", text)

    # Remove orphan punctuation surrounded by spaces
    text = re.sub(r"\s+[,:;]\s+", " ", text)

    # Remove duplicated opening title-like chunks
    words = text.split()
    if len(words) > 10:
        half = len(words) // 2
        if words[:half] == words[half:]:
            text = " ".join(words[:half])

    # Remove residual malformed parentheses
    text = re.sub(r"\(\s*\)\)+", " ", text)   # ( )) , ())) ...
    text = re.sub(r"\(\s*\)", " ", text)      # ( )

    # Remove residual formula references
    text = re.sub(r"'?\bformula_\d+\b'?", " ", text, flags=re.IGNORECASE)

    # Remove residual section numbering in quoted or malformed form
    text = re.sub(r"\b\d+(?:\.\d+){2,}\.?\b", " ", text)

    # Remove malformed Table/Figure references such as Table.3.2.3.2.1 or Fig.3
    text = re.sub(r"\b(Table|Figure|Fig)\.?\s*\d+(?:\.\d+)*\.?", " ", text, flags=re.IGNORECASE)

    # Remove complete parenthetical references
    text = re.sub(
        r"\(\s*(Fig|Figs|Figure|Figures|Table|Tables)\.?\s*[A-Za-z]?\d*(?:\.\d+)*\s*\)",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove incomplete ones like "(Fig"
    text = re.sub(
        r"\(\s*(Fig|Figs|Figure|Figures|Table|Tables)\b\.?",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove quoted figure/table references like 'fig, 'Fig., 'table
    text = re.sub(
        r"['\"]\s*(Fig|Figs|Figure|Figures|Table|Tables)\.?\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Final whitespace normalization after residual noise removal
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Fix stray quotes around punctuation
    text = re.sub(r"['\"]?\s*([.,;:])\s*['\"]?", r"\1", text)

    # Advanced punctuation normalization
    # Rules:
    # - if a punctuation group contains a dot, keep only "."
    # - otherwise compress repeated commas/semicolons/colons
    text = re.sub(r"[.,;:]*\.[.,;:]*", ".", text)
    text = re.sub(r"(,)\1+", r"\1", text)
    text = re.sub(r"(;)\1+", r"\1", text)
    text = re.sub(r"(:)\1+", r"\1", text)

    # Ensure space after punctuation (if missing)
    text = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", text)

    return text


df = pd.read_excel("all_corpus_processed.xlsx")

#keeps only the rows where "match" column is 1, which means the text segment is relevant to the query
df = df[df["match"] == 1]

df["text_segment_clean"] = df["text_segment"].apply(clean_text)

df.to_excel("cleaned_dataset.xlsx", index=False)

# Prepare LLM input
llm_input_df = df[["text_segment_clean"]].copy()

# Remove empty or very short sentences
llm_input_df = llm_input_df[llm_input_df["text_segment_clean"].str.len() > 5]

# Clean and convert to list
sentences = []
for s in llm_input_df["text_segment_clean"]:
    if isinstance(s, str):
        s = s.strip().replace("\n", " ")
        s = " ".join(s.split())  # remove extra spaces
        if len(s) > 20:  # filter out very short sentences
            sentences.append(s)

# Save as TXT
with open("datasets/lulc.txt", "w", encoding="utf-8") as f:
    for s in sentences:
        f.write(s + "\n")