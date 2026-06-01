"""Report-friendly model comparison summaries for the Bati Bank project."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import MODELS_DIR

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def build_comparison_table(all_metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Build a ranked comparison table from per-model test metrics."""
    rows = [{"model": model_name, **metrics} for model_name, metrics in all_metrics.items()]
    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    comparison.insert(0, "rank", range(1, len(comparison) + 1))

    columns = ["rank", "model", *METRIC_COLUMNS]
    return comparison[columns].round(4)


def build_best_model_summary(
    best_model_name: str,
    all_metrics: dict[str, dict[str, float]],
    *,
    selection_metric: str = "roc_auc",
) -> dict[str, object]:
    """Create a concise summary for the selected best model."""
    best_metrics = all_metrics[best_model_name]
    return {
        "best_model": best_model_name,
        "selection_metric": selection_metric,
        "selection_score": round(float(best_metrics[selection_metric]), 4),
        "metrics": {metric: round(float(value), 4) for metric, value in best_metrics.items()},
    }


def _format_markdown_table(df: pd.DataFrame) -> str:
    """Format a dataframe as a simple markdown table without extra dependencies."""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def format_comparison_markdown(
    comparison: pd.DataFrame,
    best_summary: dict[str, object],
) -> str:
    """Render a markdown snippet suitable for pasting into a final report."""
    lines = [
        "## Model Comparison Summary",
        "",
        "Test-set performance across candidate models (ranked by ROC-AUC):",
        "",
        _format_markdown_table(comparison),
        "",
        "### Best Model",
        "",
        f"- **Model:** `{best_summary['best_model']}`",
        f"- **Selected by:** `{best_summary['selection_metric']}` "
        f"= `{best_summary['selection_score']}`",
        "",
        "| Metric | Score |",
        "|--------|-------|",
    ]

    for metric, value in best_summary["metrics"].items():
        lines.append(f"| {metric} | {value} |")

    lines.append("")
    return "\n".join(lines)


def save_model_comparison_results(
    all_metrics: dict[str, dict[str, float]],
    best_model_name: str,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Save comparison table, best-model summary, and markdown report snippet.

    Outputs
    -------
    - ``model_comparison.csv`` — full metrics table for all models
    - ``best_model_summary.json`` — best model name and key metrics
    - ``model_comparison.md`` — report-ready markdown section
    """
    out_dir = Path(output_dir or MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison_table(all_metrics)
    best_summary = build_best_model_summary(best_model_name, all_metrics)

    csv_path = out_dir / "model_comparison.csv"
    json_path = out_dir / "best_model_summary.json"
    md_path = out_dir / "model_comparison.md"

    comparison.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    md_path.write_text(format_comparison_markdown(comparison, best_summary), encoding="utf-8")

    return {
        "comparison_csv": csv_path,
        "best_model_summary": json_path,
        "comparison_markdown": md_path,
    }
