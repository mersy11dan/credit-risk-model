"""Pydantic request/response schemas for the credit risk API."""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Customer-level features aligned with the trained Bati Bank model."""

    total_transaction_amount: float = Field(..., description="Sum of customer transaction amounts")
    avg_transaction_amount: float = Field(..., description="Average transaction amount")
    transaction_count: int = Field(..., ge=0, description="Number of transactions")
    std_transaction_amount: float = Field(..., ge=0, description="Std dev of transaction amounts")
    txn_hour: int = Field(..., ge=0, le=23, description="Hour of most recent transaction")
    txn_day: int = Field(..., ge=1, le=31, description="Day of most recent transaction")
    txn_month: int = Field(..., ge=1, le=12, description="Month of most recent transaction")
    txn_year: int = Field(..., ge=2000, le=2100, description="Year of most recent transaction")
    mode_ProductCategory: str = Field(..., description="Most frequent product category")
    mode_ChannelId: str | None = Field(default=None, description="Most frequent channel id")
    mode_CurrencyCode: str | None = Field(default=None, description="Most frequent currency code")
    mode_ProviderId: str | None = Field(default=None, description="Most frequent provider id")
    mode_PricingStrategy: str | None = Field(
        default=None,
        description="Most frequent pricing strategy",
    )
    mode_CountryCode: str | None = Field(default=None, description="Most frequent country code")

    def to_feature_row(self) -> dict[str, object]:
        """Return a model-input dictionary excluding unset optional fields."""
        return self.model_dump(exclude_none=True)


class PredictionResponse(BaseModel):
    """Risk scoring response for a single customer."""

    risk_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability of high proxy risk (is_high_risk=1)",
    )
    risk_category: str = Field(..., description="Low, Medium, or High risk band")


def risk_category_from_probability(probability: float) -> str:
    """Map risk probability to a simple business-facing category."""
    if probability < 0.2:
        return "Low"
    if probability < 0.5:
        return "Medium"
    return "High"
