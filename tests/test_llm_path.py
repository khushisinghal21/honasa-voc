"""Exercise the Claude code path with a fake client (no network, no key)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import voc.llm as llm_mod
from voc.pipeline import extract
from voc.schemas import AspectMention, BatchExtraction, ReviewExtraction, ThemeLabel


class FakeLLM:
    """Pretends to be Claude: skips one review on the batch call, and paraphrases one span."""

    def __init__(self):
        self.calls = []

    def structured(self, schema, prompt, system="", tier="fast", max_tokens=4096, attempts=3):
        self.calls.append(prompt)
        if schema is BatchExtraction:
            results = []
            if "r1" in prompt:
                results.append(ReviewExtraction(
                    review_id="r1", is_relevant=True, overall_sentiment="negative",
                    aspects=[
                        AspectMention(aspect="packaging", sentiment="negative", severity=2,
                                      evidence_span="pump stopped working"),            # verbatim
                        AspectMention(aspect="price_value", sentiment="negative", severity=2,
                                      evidence_span="it is far too expensive"),        # hallucinated
                    ]))
            if "r2" in prompt and len(self.calls) > 1:  # skipped on the first (batch) call, returned on retry
                results.append(ReviewExtraction(review_id="r2", is_relevant=False, overall_sentiment="neutral"))
            return BatchExtraction(results=results)
        if schema is ThemeLabel:
            return ThemeLabel(label="Pump stops working", summary="Users say the pump stops working.")
        raise AssertionError(schema)


@pytest.fixture
def fake(monkeypatch):
    f = FakeLLM()
    monkeypatch.setattr(extract, "get_llm", lambda: f)
    monkeypatch.setattr(extract, "product_map", lambda: {"p": {"name": "Test Shampoo", "match_terms": []}})
    return f


def test_llm_extraction_verifies_spans_and_retries_missing(fake):
    reviews = [
        {"review_id": "r1", "product_id": "p", "text": "The pump stopped working after a week. Pricey too."},
        {"review_id": "r2", "product_id": "p", "text": "Which camera do you use?"},
    ]
    results, stats = extract.extract_reviews(reviews, backend="anthropic")
    by_id = {r.review_id: r for r in results}
    assert set(by_id) == {"r1", "r2"}, "the skipped review must be recovered by the per-item retry"
    assert [m.aspect for m in by_id["r1"].aspects] == ["packaging"], "hallucinated span must be dropped"
    assert by_id["r1"].aspects[0].evidence_span == "pump stopped working"
    assert stats["spans_total"] == 2 and stats["spans_verified"] == 1
    assert stats["span_verification_rate"] == 0.5


def test_structured_output_retries_on_validation_error(monkeypatch):
    """LLM.structured should feed the validation error back and accept the corrected second answer."""
    monkeypatch.setattr(llm_mod.settings, "anthropic_api_key", "test")
    bad = {"label": "x"}  # missing `summary`
    good = {"label": "Pump stops working", "summary": "Consumers say the pump jams."}
    responses = iter([bad, good])
    sent = []

    def create(**kw):
        sent.append(kw["messages"])
        block = SimpleNamespace(type="tool_use", id="tu_1", input=next(responses))
        return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=10, output_tokens=5))

    client = llm_mod.LLM.__new__(llm_mod.LLM)
    client.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    out = client.structured(ThemeLabel, "label this", tier="smart")
    assert out.label == "Pump stops working"
    assert len(sent) == 2
    retry_msg = sent[1][-1]["content"][0]
    assert retry_msg["type"] == "tool_result" and retry_msg["is_error"]


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.setattr(llm_mod.settings, "anthropic_api_key", "")
    with pytest.raises(llm_mod.LLMError, match="baseline"):
        llm_mod.LLM()
