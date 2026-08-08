import pandas as pd

from src.data_processing import compute_rfm_metrics, rfm_kmeans_proxy_target


def _tx() -> pd.DataFrame:
    # Three customers with different engagement patterns.
    return pd.DataFrame(
        {
            "CustomerId": ["A", "A", "A", "B", "B", "C"],
            "TransactionStartTime": [
                "2019-01-10T00:00:00Z",
                "2019-01-11T00:00:00Z",
                "2019-01-12T00:00:00Z",
                "2019-01-05T00:00:00Z",
                "2019-01-06T00:00:00Z",
                "2018-12-01T00:00:00Z",
            ],
            "Value": [100, 120, 90, 10, 15, 5],
        }
    )


def test_compute_rfm_uses_consistent_snapshot_date():
    df = _tx()
    snap = pd.Timestamp("2019-01-13", tz="UTC")
    rfm = compute_rfm_metrics(df, snapshot_date=snap)

    assert set(rfm.columns) == {"CustomerId", "recency_days", "frequency", "monetary"}
    assert len(rfm) == 3

    # Customer A last txn 2019-01-12 -> recency ~1 day
    rec_a = float(rfm.loc[rfm["CustomerId"] == "A", "recency_days"].iloc[0])
    assert 0.9 <= rec_a <= 1.1


def test_rfm_kmeans_proxy_target_outputs_binary_label():
    df = _tx()
    snap = pd.Timestamp("2019-01-13", tz="UTC")
    proxy = rfm_kmeans_proxy_target(df, snapshot_date=snap, random_state=42)

    assert {"CustomerId", "rfm_cluster", "is_high_risk"} <= set(proxy.columns)
    assert set(proxy["is_high_risk"].unique().tolist()) <= {0, 1}
    assert len(proxy) == 3
