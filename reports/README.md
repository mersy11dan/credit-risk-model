# Reports – Bati Bank Credit Risk Model

## Files

| File | Description |
|------|-------------|
| `Interim_Report.md` / `.html` | Interim submission |
| **`final_report.md`** | **Final report (27 sections, publication quality)** |
| **`final_report.html`** | **Print-ready final report with figures** |
| `figures/` | Interim report charts |
| `figures_final/` | **22 final report charts** |
| `analysis_results.json` | EDA/RFM numeric results used in final report |
| `feature_importance.json` | GB model feature importances |

## Regenerate figures

```bash
set PYTHONPATH=.
python scripts/generate_report_figures.py      # interim figures
python scripts/generate_final_report_figures.py  # final report figures
```

## Export final report to PDF

1. Open **`final_report.html`** in Chrome or Edge.
2. **Ctrl+P** → **Save as PDF** → enable **Background graphics**.
3. Save as `final_report.pdf`.

## Training artifacts (for report metrics)

```bash
python -m src.train
```

Produces `models/model_comparison.csv`, `models/best_model_summary.json`, `mlruns/`.
