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
    [1.0000, 0.7097, 0.4355, 0.3226],

    # GPT5.5 ->
    [0.4037, 1.0000, 0.3119, 0.2294],

    # DeepSeek R1 ->
    [0.5870, 0.7391, 1.0000, 0.3913],

    # LLaMa3 ->
    [0.4444, 0.5556, 0.4000, 1.0000],
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
# Mean Local Jaccard
# =========================================

jaccard_matrix = [
    # GPT4o-mini
    [1.0000, 0.6519, 0.6423, 0.6645],

    # GPT5.5
    [0.6519, 1.0000, 0.6554, 0.5643],

    # DeepSeekR1
    [0.6423, 0.6554, 1.0000, 0.6383],

    # LLaMa3
    [0.6645, 0.5643, 0.6383, 1.0000],
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
    cbar_kws={"label": "Mean Local Jaccard"}
)

plt.xlabel("Model")
plt.ylabel("Model")

plt.figtext(
    0.5,
    0.01,
    "Each cell represents the average sentence-level Jaccard similarity between two models.",
    ha="center",
    fontsize=10
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

plt.show()