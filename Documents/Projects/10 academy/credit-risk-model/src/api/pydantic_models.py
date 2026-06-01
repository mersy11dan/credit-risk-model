"""Pydantic request/response schemas for the credit risk API."""

from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    """Single loan application payload for PD scoring."""

    age: int = Field(..., ge=18, description="Applicant age in years")
    income: float = Field(..., gt=0, description="Annual income")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_term_months: int = Field(..., gt=0, description="Loan term in months")
    credit_score: int = Field(..., ge=300, le=850, description="Credit bureau score")
    employment_years: float = Field(..., ge=0, description="Years in current employment")


class PredictionResponse(BaseModel):
    """Probability-of-default prediction response."""

    probability_of_default: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that the applicant will default",
    )
    risk_category: str = Field(..., description="Low, Medium, or High risk band")


def risk_category_from_probability(probability: float) -> str:
    """Map PD score to a simple risk band for business users."""
    if probability < 0.2:
        return "Low"
    if probability < 0.5:
        return "Medium"
    return "High"
