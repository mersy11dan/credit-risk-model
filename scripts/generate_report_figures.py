"""Generate figures for the Bati Bank interim report."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")

    if not DATA_PATH.exists():
        print(f"Data not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce", utc=True)

    # Figure 1: Transaction value distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    positive = df[df["Value"] > 0]["Value"]
    sns.histplot(positive, bins=50, kde=True, ax=ax, color="steelblue")
    ax.set_title("Distribution of Transaction Values (Value > 0)")
    ax.set_xlabel("Transaction value (UGX)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_transaction_value_distribution.png", dpi=150)
    plt.close(fig)

    # Figure 2: Product category volume
    fig, ax = plt.subplots(figsize=(8, 4))
    order = df["ProductCategory"].value_counts().head(8).index
    sns.countplot(data=df, y="ProductCategory", order=order, ax=ax, color="coral")
    ax.set_title("Transaction Volume by Product Category")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_product_category_volume.png", dpi=150)
    plt.close(fig)

    # Figure 3: Proxy event rate by product
    fraud = (
        df.groupby("ProductCategory")["FraudResult"]
        .agg(proxy_rate="mean", count="count")
        .reset_index()
        .sort_values("proxy_rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=fraud, x="ProductCategory", y="proxy_rate", ax=ax, color="indianred")
    ax.set_title("Fraud Flag Rate by Product Category (Behavioral Proxy)")
    ax.set_ylabel("Proxy event rate")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_proxy_rate_by_product.png", dpi=150)
    plt.close(fig)

    # Figure 4: Channel distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    channel = df["ChannelId"].value_counts().head(6)
    channel.plot(kind="bar", ax=ax, color="seagreen")
    ax.set_title("Transactions by Channel")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_transactions_by_channel.png", dpi=150)
    plt.close(fig)

    # Figure 5: RFM proxy (if pipeline runs)
    try:
        from src.data_processing import rfm_kmeans_proxy_target

        proxy = rfm_kmeans_proxy_target(df)
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(
            proxy["frequency"],
            proxy["recency_days"],
            c=proxy["is_high_risk"],
            cmap="coolwarm",
            alpha=0.7,
            s=proxy["monetary"] / proxy["monetary"].max() * 200 + 20,
        )
        ax.set_xlabel("Transaction frequency")
        ax.set_ylabel("Recency (days)")
        ax.set_title("RFM Customer Segments (size = monetary, color = is_high_risk)")
        fig.colorbar(scatter, ax=ax, label="is_high_risk")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "05_rfm_proxy_clusters.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Skipped RFM figure: {exc}")

    # Figure 6: Project pipeline diagram
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    steps = [
        "Raw Data",
        "EDA",
        "Feature Eng.",
        "RFM Proxy",
        "Train Models",
        "MLflow",
        "FastAPI",
        "Docker",
    ]
    for i, step in enumerate(steps):
        ax.text(i * 1.2, 0.5, step, ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round", facecolor="#ddeeff"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(i * 1.2 + 0.55, 0.5), xytext=(i * 1.2 + 0.35, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_xlim(-0.2, len(steps) * 1.2)
    ax.set_ylim(0, 1)
    ax.set_title("Bati Bank Credit Risk Pipeline (Interim)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_project_pipeline.png", dpi=150)
    plt.close(fig)

    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
