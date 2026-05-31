"""Pydantic schemas for the credit risk API."""

from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    age: int = Field(..., ge=18)
    income: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., ge=0, le=1)
