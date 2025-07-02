import pytest
from app.core.moderation import GuardianModerationEngine

@pytest.fixture(scope="module")
def moderation_engine():
    return GuardianModerationEngine()

def test_positive_scammy_financial_post_flagged(moderation_engine):
    # Example of a highly positive, scammy financial post
    scammy_examples = [
        "This is a guaranteed return! Invest now for unbelievable profits!",
        "Get rich quick with this exclusive offer. Financial freedom awaits!",
        "Double your money, no risk, act now!",
        "Make $1000 per day, risk-free, join our team!",
        "Click here to invest and get secret investment tips!"
    ]
    for example in scammy_examples:
        result = moderation_engine.stage3_finbert_check(example)
        assert not result.accepted, f"Should not accept: {example}"
        assert result.action == "FLAG_HIGH", f"Should flag as high: {example}"
        assert "scammy" in result.reason or "promotional" in result.reason

def test_positive_non_scammy_financial_post_passes(moderation_engine):
    # Example of a positive, non-scammy financial post
    safe_examples = [
        "I am happy with my investment returns this year.",
        "Our company had a great quarter and we are optimistic about the future.",
        "Congratulations to all investors for the positive results!"
    ]
    for example in safe_examples:
        result = moderation_engine.stage3_finbert_check(example)
        # Should not be flagged or blocked
        assert result.accepted, f"Should accept: {example}"
        assert result.action in ("PASS", "FLAG_LOW", "FLAG_MEDIUM", "FLAG_HIGH"), f"Unexpected action: {result.action}" 