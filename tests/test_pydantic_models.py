"""Tests for Pydantic API schemas."""

from src.api.pydantic_models import risk_category_from_probability


def test_risk_category_low():
    assert risk_category_from_probability(0.1) == "Low"


def test_risk_category_medium():
    assert risk_category_from_probability(0.3) == "Medium"


def test_risk_category_high():
    assert risk_category_from_probability(0.7) == "High"
