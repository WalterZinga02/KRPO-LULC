from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
INPUT_FILE = OUTPUT_DIR / "triple_matching_analysis.xlsx"
SUMMARY_SHEET = "Summary"


def load_summary() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}. Run metrics/match_checker.py first."
        )

    return pd.read_excel(INPUT_FILE, sheet_name=SUMMARY_SHEET)


def model_order(summary_df: pd.DataFrame) -> list[str]:
    return sorted(set(summary_df["model_a"]).union(summary_df["model_b"]))


def build_overlap_matrix(summary_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=models, columns=models)

    for model in models:
        matrix.loc[model, model] = 1.0

    for _, row in summary_df.iterrows():
        model_a = row["model_a"]
        model_b = row["model_b"]
        matrix.loc[model_a, model_b] = row["overlap_on_a"]
        matrix.loc[model_b, model_a] = row["overlap_on_b"]

    return matrix


def build_jaccard_matrix(summary_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0.0, index=models, columns=models)

    for model in models:
        matrix.loc[model, model] = 1.0

    for _, row in summary_df.iterrows():
        model_a = row["model_a"]
        model_b = row["model_b"]
        score = row["mean_local_jaccard"]
        matrix.loc[model_a, model_b] = score
        matrix.loc[model_b, model_a] = score

    return matrix


def plot_heatmap(
    df: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    footnote: str,
    cmap: str,
    filename: str,
) -> None:
    size = max(8, len(df) * 1.25)
    fig, ax = plt.subplots(figsize=(size, size - 1))

    image = ax.imshow(df.values, cmap=cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)

    ax.set_title(title, fontsize=15, pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=35, ha="right")
    ax.set_yticklabels(df.index)

    ax.set_xticks(np.arange(-0.5, len(df.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(len(df.index)):
        for col_idx in range(len(df.columns)):
            value = df.iat[row_idx, col_idx]
            text_color = "white" if value >= 0.55 else "#222222"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=11,
            )

    fig.text(
        0.5,
        0.01,
        footnote,
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def main() -> None:
    summary_df = load_summary()
    models = model_order(summary_df)

    overlap_df = build_overlap_matrix(summary_df, models)
    jaccard_df = build_jaccard_matrix(summary_df, models)

    plot_heatmap(
        df=overlap_df,
        title="Pairwise Semantic Overlap",
        xlabel="Target Model",
        ylabel="Source Model",
        colorbar_label="Semantic Overlap",
        footnote=(
            "Each cell represents the percentage of triples from the row model "
            "that match triples from the column model."
        ),
        cmap="Blues",
        filename="semantic_overlap_heatmap.png",
    )

    plot_heatmap(
        df=jaccard_df,
        title="Pairwise Mean Local Jaccard",
        xlabel="Model",
        ylabel="Model",
        colorbar_label="Mean Local Jaccard",
        footnote=(
            "Each cell represents the average sentence-level Jaccard similarity "
            "between two models."
        ),
        cmap="Greens",
        filename="mean_local_jaccard_heatmap.png",
    )


if __name__ == "__main__":
    main()
