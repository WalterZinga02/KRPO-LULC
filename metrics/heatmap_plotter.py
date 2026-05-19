import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =========================================
# MODELS
# =========================================

models = [
    "GPT4o-mini",
    "GPT5.5",
    "GPT5-mini",
    "LLaMa3"
]


# =========================================
# OVERLAP MATRIX
# Rows = source model
# Columns = target model
# =========================================

matrix = [

    # GPT4o-mini ->
    [1.0000, 0.5096, 0.2682, 0.3985],

    # GPT5.5 ->
    [0.3093, 1.0000, 0.3081, 0.3349],

    # GPT5-mini ->
    [0.3211, 0.6078, 1.0000, 0.3509],

    # LLaMa3 ->
    [0.3200, 0.4431, 0.2354, 1.0000],
]


# =========================================
# DATAFRAME
# =========================================

df = pd.DataFrame(
    matrix,
    index=models,
    columns=models
)


# =========================================
# HEATMAP
# =========================================

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

#plt.title("Semantic Overlap Between Models")
plt.xlabel("Target Model")
plt.ylabel("Source Model")

plt.tight_layout()

plt.show()