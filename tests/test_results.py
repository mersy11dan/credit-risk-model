"""Tests for model comparison result summaries."""

import json
from pathlib import Path

from src.results import (
    build_best_model_summary,
    build_comparison_table,
    save_model_comparison_results,
)


def test_build_comparison_table_ranks_by_roc_auc():
    all_metrics = {
        "logistic_regression": {
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.65,
            "roc_auc": 0.75,
        },
        "random_forest": {
            "accuracy": 0.85,
            "precision": 0.8,
            "recall": 0.7,
            "f1": 0.75,
            "roc_auc": 0.9,
        },
    }

    table = build_comparison_table(all_metrics)

    expected_columns = ["rank", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    assert list(table.columns) == expected_columns
    assert table.iloc[0]["model"] == "random_forest"
    assert table.iloc[0]["rank"] == 1


def test_save_model_comparison_results_writes_report_files(tmp_path: Path):
    all_metrics = {
        "decision_tree": {
            "accuracy": 0.7,
            "precision": 0.6,
            "recall": 0.5,
            "f1": 0.55,
            "roc_auc": 0.65,
        },
        "gradient_boosting": {
            "accuracy": 0.82,
            "precision": 0.78,
            "recall": 0.72,
            "f1": 0.75,
            "roc_auc": 0.88,
        },
    }

    paths = save_model_comparison_results(
        all_metrics,
        best_model_name="gradient_boosting",
        output_dir=tmp_path,
    )

    assert paths["comparison_csv"].exists()
    assert paths["best_model_summary"].exists()
    assert paths["comparison_markdown"].exists()

    summary = json.loads(paths["best_model_summary"].read_text(encoding="utf-8"))
    assert summary["best_model"] == "gradient_boosting"
    assert summary["selection_metric"] == "roc_auc"
    assert summary["metrics"]["f1"] == 0.75

    markdown = paths["comparison_markdown"].read_text(encoding="utf-8")
    assert "## Model Comparison Summary" in markdown
    assert "gradient_boosting" in markdown


def test_build_best_model_summary_rounds_metrics():
    summary = build_best_model_summary(
        "logistic_regression",
        {
            "logistic_regression": {
                "accuracy": 0.812345,
                "precision": 0.712345,
                "recall": 0.612345,
                "f1": 0.652345,
                "roc_auc": 0.752345,
            }
        },
    )

    assert summary["best_model"] == "logistic_regression"
    assert summary["selection_score"] == 0.7523
    assert summary["metrics"]["accuracy"] == 0.8123
