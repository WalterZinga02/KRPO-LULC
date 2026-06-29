from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
BENCHMARK_FILE = OUTPUT_DIR / "benchmark_comparison.xlsx"

MODEL_LABELS = {
    "deepseek-r1:8b": "DeepSeek-R1",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemma3:27b": "Gemma 3",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-5.5": "GPT-5.5",
    "llama3:8b": "LLaMA3",
    "qwen3:14b": "Qwen3",
}

MODEL_COLORS = {
    "DeepSeek-R1": "#6C5B7B",
    "Gemini 2.5 Flash": "#4C78A8",
    "Gemma 3": "#72B7B2",
    "GPT-4o-mini": "#59A14F",
    "GPT-5.5": "#F28E2B",
    "LLaMA3": "#B07AA1",
    "Qwen3": "#E15759",
}

GEMINI_25_FLASH_INPUT_PRICE_PER_1M = 0.30
GEMINI_25_FLASH_OUTPUT_PRICE_PER_1M = 2.50


def load_sentence_metrics() -> pd.DataFrame:
    df = pd.read_excel(BENCHMARK_FILE, sheet_name="normalized_per_sentence")
    df["model"] = df["model"].map(MODEL_LABELS).fillna(df["model"])

    gemini_mask = df["model"] == "Gemini 2.5 Flash"
    df.loc[gemini_mask, "api_cost_usd_per_sentence"] = (
        df.loc[gemini_mask, "prompt_tokens_per_sentence"]
        * GEMINI_25_FLASH_INPUT_PRICE_PER_1M
        / 1_000_000
        + df.loc[gemini_mask, "completion_tokens_per_sentence"]
        * GEMINI_25_FLASH_OUTPUT_PRICE_PER_1M
        / 1_000_000
    )
    df["api_cost_usd_per_sentence"] = df["api_cost_usd_per_sentence"].fillna(0)

    return df


def plot_metric_grid(df: pd.DataFrame, metrics: list[str], title: str, filename: str) -> None:
    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 4.8 * nrows))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        plot_df = df[["model", metric]].copy()
        if metric != "api_cost_usd_per_sentence":
            plot_df = plot_df.dropna()

        colors = [
            MODEL_COLORS.get(model, "#B3B3B3")
            for model in plot_df["model"]
        ]

        ax.bar(
            plot_df["model"],
            plot_df[metric],
            color=colors,
            edgecolor="#2F2F2F",
            linewidth=0.6,
        )

        ax.set_title(metric.replace("_", " "), fontsize=11)
        ax.set_ylabel(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=25)

        values = plot_df[metric]
        if len(values) > 0 and values.min() > 0:
            ratio = values.max() / values.min()
            if ratio > 1000:
                ax.set_yscale("log")
                ax.set_ylabel(metric.replace("_", " ") + " (log scale)")

        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_dir = OUTPUT_DIR / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


sentence_metrics = [
    "runtime_sec_per_sentence",
    "total_tokens_per_sentence",
    "energy_kwh_per_sentence",
    "co2_kg_per_sentence",
    "api_cost_usd_per_sentence",
    "final_triplets_per_sentence",
]


df_sentence = load_sentence_metrics()

plot_metric_grid(
    df=df_sentence,
    metrics=sentence_metrics,
    title="Normalized benchmarking metrics per processed sentence",
    filename="benchmark_per_sentence.png",
)

print("Done.")
