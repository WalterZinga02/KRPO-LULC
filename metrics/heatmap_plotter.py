import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


models = [
    "GPT4o-mini",
    "GPT5.5",
    "DeepSeekR1",
    "LLaMa3"
]


matrix = [

    # GPT4o-mini ->
    [1.0000, 0.5370, 0.3287, 0.2269],

    # GPT5.5 ->
    [0.2739, 1.0000, 0.2869, 0.1747],

    # DeepSeek R1 ->
    [0.4595, 0.7864, 1.0000, 0.2816],

    # LLaMa3 ->
    [0.2579, 0.3895, 0.2289, 1.0000],
]


df = pd.DataFrame(
    matrix,
    index=models,
    columns=models
)


plt.figure(figsize=(9, 7))

sns.heatmap(
    df,
    annot=True,
    cmap="Blues",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    square=True,
    fmt=".2f",
    cbar_kws={"label": "Semantic Overlap"}
)

plt.xlabel("Target Model")
plt.ylabel("Source Model")

plt.figtext(
    0.5,
    0.01,
    "Each cell represents the percentage of triples from the row model that match triples from the column model.",
    ha="center",
    fontsize=10
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()


# =========================================
# JACCARD MATRIX
# =========================================

jaccard_matrix = [

    # GPT4o-mini
    [1.0000, 0.2216, 0.2379, 0.1370],

    # GPT5.5
    [0.2216, 1.0000, 0.2662, 0.1375],

    # DeepSeekR1
    [0.2379, 0.2662, 1.0000, 0.1445],

    # LLaMa3
    [0.1370, 0.1375, 0.1445, 1.0000],
]


# =========================================
# DATAFRAME
# =========================================

jaccard_df = pd.DataFrame(
    jaccard_matrix,
    index=models,
    columns=models
)


# =========================================
# JACCARD HEATMAP
# =========================================

plt.figure(figsize=(9, 7))

sns.heatmap(
    jaccard_df,
    annot=True,
    cmap="Greens",
    vmin=0,
    vmax=1,
    linewidths=0.5,
    square=True,
    fmt=".2f",
    cbar_kws={"label": "Jaccard Similarity"}
)

plt.xlabel("Model")
plt.ylabel("Model")

plt.figtext(
    0.5,
    0.01,
    "Jaccard similarity = shared triples / total unique triples between two models.",
    ha="center",
    fontsize=10
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

plt.show()