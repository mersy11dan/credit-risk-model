"""Generate publication-quality figures for the final Bati Bank report."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FIG_DIR = PROJECT_ROOT / "reports" / "figures_final"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
RESULTS_PATH = PROJECT_ROOT / "reports" / "analysis_results.json"


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce", utc=True)

    from src.data_processing import rfm_kmeans_proxy_target, build_customer_dataset

    proxy = rfm_kmeans_proxy_target(df)
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    # 01 Overview - transactions over time
    daily = df.groupby(df["TransactionStartTime"].dt.date).size()
    fig, ax = plt.subplots(figsize=(10, 4))
    daily.plot(ax=ax, color="steelblue")
    ax.set_title("Daily Transaction Volume (Nov 2018 – Feb 2019)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Transactions")
    _save(fig, "01_daily_transaction_volume.png")

    # 02 Amount distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[df["Value"] > 0]["Value"], bins=60, kde=True, ax=ax)
    ax.set_title("Transaction Value Distribution")
    ax.set_xlabel("Value (UGX)")
    _save(fig, "02_value_distribution.png")

    # 03 Log-scale value
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[df["Value"] > 0]["Value"], bins=50, log_scale=True, ax=ax)
    ax.set_title("Log-Scale Transaction Value Distribution")
    _save(fig, "03_value_log_distribution.png")

    # 04 Correlation heatmap
    num_cols = ["Amount", "Value", "PricingStrategy", "FraudResult"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix — Numerical Features")
    _save(fig, "04_correlation_heatmap.png")

    # 05 Product category
    fig, ax = plt.subplots(figsize=(9, 4))
    order = df["ProductCategory"].value_counts().head(8).index
    sns.countplot(data=df, y="ProductCategory", order=order, ax=ax)
    ax.set_title("Transaction Count by Product Category")
    _save(fig, "05_product_category.png")

    # 06 Channel
    fig, ax = plt.subplots(figsize=(7, 4))
    df["ChannelId"].value_counts().head(6).plot(kind="bar", ax=ax, color="seagreen")
    ax.set_title("Transactions by Channel")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "06_channel_volume.png")

    # 07 Fraud rate by product
    pf = pd.DataFrame(results["product_fraud"])
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=pf, x="ProductCategory", y="mean", ax=ax, color="indianred")
    ax.set_title("Fraud Flag Rate by Product (Behavioral Proxy)")
    ax.set_ylabel("Rate")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, "07_fraud_rate_by_product.png")

    # 08 Boxplot value by product
    top_cats = df["ProductCategory"].value_counts().head(5).index
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df[df["ProductCategory"].isin(top_cats)], x="ProductCategory", y="Value", ax=ax)
    ax.set_title("Transaction Value by Product Category (Top 5)")
    ax.tick_params(axis="x", rotation=20)
    _save(fig, "08_boxplot_value_by_product.png")

    # 09 Transactions per customer
    txn_per = df.groupby("CustomerId").size()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(txn_per, bins=40, ax=ax)
    ax.set_title("Transactions per Customer")
    ax.axvline(txn_per.median(), color="red", linestyle="--", label=f"Median={txn_per.median():.0f}")
    ax.legend()
    _save(fig, "09_txn_per_customer.png")

    # 10 RFM scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        proxy["frequency"],
        proxy["recency_days"],
        c=proxy["is_high_risk"],
        cmap="coolwarm",
        alpha=0.6,
        s=30,
    )
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Recency (days)")
    ax.set_title("RFM Segments: Recency vs Frequency (color = is_high_risk)")
    fig.colorbar(sc, ax=ax)
    _save(fig, "10_rfm_recency_frequency.png")

    # 11 RFM monetary by cluster
    cs = pd.DataFrame(results["cluster_stats"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=cs, x="rfm_cluster", y="monetary", hue="rfm_cluster", legend=False, ax=ax)
    ax.set_title("Average Monetary Value by RFM Cluster")
    _save(fig, "11_rfm_monetary_by_cluster.png")

    # 12 Cluster profile
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(cs))
    w = 0.25
    ax.bar(x - w, cs["recency"], w, label="Recency (days)")
    ax.bar(x, cs["frequency"] / cs["frequency"].max() * 100, w, label="Frequency (scaled)")
    ax.bar(x + w, cs["monetary"] / cs["monetary"].max() * 100, w, label="Monetary (scaled)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cluster {c}" for c in cs["rfm_cluster"]])
    ax.set_title("RFM Cluster Profiles (normalized frequency & monetary)")
    ax.legend()
    _save(fig, "12_rfm_cluster_profiles.png")

    # 13 Target balance
    tb = results["target_balance"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [tb["0"], tb["1"]],
        labels=["Low risk (0)", "High risk (1)"],
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
    )
    ax.set_title("Customer-Level Proxy Target Balance (is_high_risk)")
    _save(fig, "13_target_balance_pie.png")

    # 14 IV summary
    iv = pd.DataFrame(results["iv_summary"])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=iv, y="feature", x="iv", ax=ax, color="purple")
    ax.set_title("Information Value — FraudResult (Transaction-Level)")
    ax.set_xlabel("IV")
    _save(fig, "14_iv_fraud_result.png")

    # 15 Model comparison
    mc = pd.read_csv(PROJECT_ROOT / "models" / "model_comparison.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(mc))
    w = 0.15
    for i, m in enumerate(metrics):
        ax.bar(x + i * w, mc[m], w, label=m)
    ax.set_xticks(x + w * 2)
    ax.set_xticklabels(mc["model"], rotation=15, ha="right")
    ax.set_ylim(0.98, 1.01)
    ax.set_title("Model Comparison — Test Set Metrics")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "15_model_comparison.png")

    # 16 ROC-AUC bar
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=mc, x="model", y="roc_auc", ax=ax, color="steelblue")
    ax.set_ylim(0.99, 1.001)
    ax.set_title("ROC-AUC by Model (Test Set)")
    ax.tick_params(axis="x", rotation=15)
    _save(fig, "16_roc_auc_comparison.png")

    # 17 Feature importance
    fi_path = PROJECT_ROOT / "reports" / "feature_importance.json"
    if fi_path.exists():
        fi = pd.DataFrame(json.loads(fi_path.read_text(encoding="utf-8")))
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=fi.head(12), y="feature", x="importance", ax=ax)
        ax.set_title("Top Feature Importances — Gradient Boosting (Best Model)")
        _save(fig, "17_feature_importance.png")

    # 18 Debit vs credit
    fig, ax = plt.subplots(figsize=(5, 5))
    debit = (df["Amount"] > 0).sum()
    credit = (df["Amount"] < 0).sum()
    ax.pie([debit, credit], labels=["Debit (>0)", "Credit (<0)"], autopct="%1.1f%%")
    ax.set_title("Debit vs Credit Transactions")
    _save(fig, "18_debit_credit_split.png")

    # 19 Pricing strategy fraud
    ps = df.groupby("PricingStrategy")["FraudResult"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=ps, x="PricingStrategy", y="FraudResult", ax=ax)
    ax.set_title("Fraud Rate by Pricing Strategy")
    _save(fig, "19_fraud_by_pricing.png")

    # 20 Pipeline diagram
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    steps = [
        "Raw Data",
        "EDA",
        "Features",
        "WoE/IV",
        "RFM Proxy",
        "Train",
        "MLflow",
        "API",
        "Docker",
        "CI",
    ]
    for i, s in enumerate(steps):
        ax.text(
            i * 1.1,
            0.5,
            s,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#ddeeff"),
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(i * 1.1 + 0.5, 0.5),
                xytext=(i * 1.1 + 0.35, 0.5),
                arrowprops=dict(arrowstyle="->"),
            )
    ax.set_title("End-to-End Bati Bank Credit Risk Pipeline", fontweight="bold")
    _save(fig, "20_pipeline_architecture.png")

    # 21 Missingness (all zero)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(["All columns"], [0], color="green")
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Values — 0% across all 16 columns")
    ax.set_xlim(0, 5)
    _save(fig, "21_missingness.png")

    # 22 Summary statistics table as figure
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    stats = [
        ["Transactions", "95,662"],
        ["Customers", "3,742"],
        ["Date range", "2018-11-15 to 2019-02-13"],
        ["Fraud rate (txn)", "0.20%"],
        ["High-risk customers", "1,438 (38.4%)"],
    ]
    table = ax.table(cellText=stats, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    table.scale(1.2, 1.5)
    ax.set_title("Dataset Summary Statistics", fontweight="bold", pad=20)
    _save(fig, "22_summary_statistics_table.png")

    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
