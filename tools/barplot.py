import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

# =========================
# DATA
# =========================

df_triplet = pd.DataFrame({
    "model": ["GPT-4o-mini", "GPT-5.5", "LLaMA3", "DeepSeek-R1"],
    "runtime_sec_per_final_triplet": [0.888414159, 2.883055444, 0.545026721, 24.78913634],
    "prompt_tokens_per_final_triplet": [923.7523148, 465.1145218, 1051.523684, 1357.857605],
    "completion_tokens_per_final_triplet": [21.34027778, 121.5442739, 42.47894737, 2568.708738],
    "total_tokens_per_final_triplet": [945.0925926, 586.6587957, 1094.002632, 3926.566343],
    "energy_kwh_per_final_triplet": [1.83694E-06, None, 6.28374E-05, 0.003678386],
    "co2_kg_per_final_triplet": [8.16345E-07, None, 3.52134E-06, 0.000206133],
    "api_cost_usd_per_final_triplet": [0.000151367, 0.005971901, 0, 0],
    "final_triplets_per_kwh": [544384.0619, None, 15914.09937, 271.8583395],
    "final_triplets_per_kg_co2": [1224972.017, None, 283982.5724, 4851.234666]
})

df_sentence = pd.DataFrame({
    "model": ["GPT-4o-mini", "GPT-5.5", "LLaMA3", "DeepSeek-R1"],
    "runtime_sec_per_sentence": [1.279316389, 8.139826537, 0.690367181, 25.53281043],
    "prompt_tokens_per_sentence": [1330.203333, 1313.173333, 1331.93, 1398.593333],
    "completion_tokens_per_sentence": [30.73, 343.16, 53.80666667, 2645.77],
    "total_tokens_per_sentence": [1360.933333, 1656.333333, 1385.736667, 4044.363333],
    "energy_kwh_per_sentence": [2.64519E-06, None, 7.9594E-05, 0.003788738],
    "co2_kg_per_sentence": [1.17554E-06, None, 4.46037E-06, 0.000212317],
    "api_cost_usd_per_sentence": [0.000217969, 0.016860667, 0, 0],
    "raw_triplets_per_sentence": [2.04, 3.18, 1.973333333, 1.2],
    "final_triplets_per_sentence": [1.44, 2.823333333, 1.266666667, 1.03]
})

MODEL_COLORS = {
    "GPT-4o-mini": "#8DA0CB",   # muted blue
    "GPT-5.5": "#FC8D62",       # muted orange
    "LLaMA3": "#66C2A5",        # muted teal
    "DeepSeek-R1": "#E78AC3"    # muted pink
}

# =========================
# OUTPUT DIR
# =========================

output_dir = Path("plots")
output_dir.mkdir(exist_ok=True)

# =========================
# PLOT FUNCTION
# =========================

def plot_metric_grid(df, metrics, title, filename):
    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 4.8 * nrows))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        plot_df = df[["model", metric]].dropna()

        colors = [
            MODEL_COLORS[m]
            for m in plot_df["model"]
        ]

        ax.bar(
            plot_df["model"],
            plot_df[metric],
            color=colors,
            edgecolor="black",
            linewidth=0.8
        )

        ax.set_title(metric.replace("_", " "), fontsize=11)
        ax.set_ylabel(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=20)

        values = plot_df[metric]

        if len(values) > 0 and values.min() > 0:
            ratio = values.max() / values.min()

            if ratio > 1000:
                ax.set_yscale("log")
                ax.set_ylabel(metric.replace("_", " ") + " (log scale)")

        ax.grid(axis="y", alpha=0.3)

    # spegne eventuali subplot vuoti
    for ax in axes[len(metrics):]:
        ax.axis("off")

    # legenda unica per tutta la figura
    legend_elements = [
        Patch(facecolor=color, label=model)
        for model, color in MODEL_COLORS.items()
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02)
    )

    fig.suptitle(title, fontsize=16, y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")

# =========================
# SELECTED METRICS
# =========================

triplet_metrics = [
    "runtime_sec_per_final_triplet",
    "total_tokens_per_final_triplet",
    "energy_kwh_per_final_triplet",
    "co2_kg_per_final_triplet",
    "api_cost_usd_per_final_triplet",
]

sentence_metrics = [
    "runtime_sec_per_sentence",
    "total_tokens_per_sentence",
    "energy_kwh_per_sentence",
    "co2_kg_per_sentence",
    "api_cost_usd_per_sentence",
    #"raw_triplets_per_sentence",
    "final_triplets_per_sentence",
]


# =========================
# GENERATE FIGURES
# =========================

plot_metric_grid(
    df=df_triplet,
    metrics=triplet_metrics,
    title="Normalized benchmarking metrics per final extracted triplet",
    filename="benchmark_per_final_triplet.png"
)

plot_metric_grid(
    df=df_sentence,
    metrics=sentence_metrics,
    title="Normalized benchmarking metrics per processed sentence",
    filename="benchmark_per_sentence.png"
)

print("Done.")