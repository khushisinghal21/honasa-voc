"""Structured-output contracts. The LLM must fill these exactly; anything else is rejected."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ASPECTS = (
    "efficacy",        # does it do what it claims (hair fall, oil control, tan removal, protection)
    "skin_reaction",   # breakouts, rashes, itching, burning, dryness, irritation, allergic response
    "fragrance",
    "texture",         # consistency, white cast, greasiness, lather, feel
    "packaging",       # pump, cap, leakage, bottle, tube, size
    "price_value",
    "authenticity",    # fake/duplicate/expired product, tampered seal
    "delivery",        # shipping, damaged in transit, quick-commerce experience
)
Aspect = Literal[
    "efficacy", "skin_reaction", "fragrance", "texture",
    "packaging", "price_value", "authenticity", "delivery",
]
Sentiment = Literal["positive", "negative", "neutral"]


class AspectMention(BaseModel):
    aspect: Aspect
    sentiment: Sentiment
    severity: int = Field(
        ge=1, le=3,
        description="1 = minor/cosmetic, 2 = clear dissatisfaction, 3 = safety or trust issue "
                    "(allergic reaction, burning, suspected fake/expired product)",
    )
    evidence_span: str = Field(
        description="An EXACT, contiguous, verbatim substring copied from the review text that supports "
                    "this aspect. Do not paraphrase, translate, or fix typos.",
    )


class ReviewExtraction(BaseModel):
    review_id: str
    is_relevant: bool = Field(
        description="True only if the text expresses a first-hand or specific opinion about THIS product. "
                    "False for spam, questions to the creator, generic chatter, or other products.",
    )
    overall_sentiment: Sentiment
    aspects: list[AspectMention] = Field(default_factory=list)


class BatchExtraction(BaseModel):
    results: list[ReviewExtraction]


class ThemeLabel(BaseModel):
    label: str = Field(description="Specific, plain-English theme name, max 8 words, e.g. 'Pump stops working mid-bottle'")
    summary: str = Field(description="One sentence describing what these consumers report. Only claims supported by the quotes.")


class RoutingDecision(BaseModel):
    team: str
    rationale: str = Field(description="One sentence")


class JudgeVerdict(BaseModel):
    supported: bool = Field(description="True if every claim in the label and summary is supported by the evidence")
    unsupported_claims: list[str] = Field(default_factory=list)
    score: float = Field(ge=0, le=1, description="Fraction of claims supported")


class SQLAnswer(BaseModel):
    sql: str
