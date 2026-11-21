
from src.services.review_analyzer import (
    ReviewAnalysis,
    analyze_one_review,
)


def valid_analysis():
    return ReviewAnalysis(
        sentiment="negative",
        category="delivery",
        urgency="high",
        language="English",
        summary="The order arrived late and the food was cold.",
        suggested_reply=(
            "We're sorry your order arrived late and cold. "
            "Please contact our support team so we can assist you."
        ),
    )


class FakeSuccessfulChain:
    def invoke(self, payload):
        return valid_analysis()


class FakeFailingChain:
    def invoke(self, payload):
        raise RuntimeError("Simulated provider failure")


def sample_review():
    return {
        "review_id": "R001",
        "date": "2026-11-15",
        "platform": "Google",
        "rating": 1,
        "review_text": "My order arrived very late and cold.",
    }


def test_valid_analysis_uses_allowed_values():
    result = valid_analysis()

    assert result.sentiment in {"positive", "neutral", "negative"}
    assert result.category in {
        "food_quality",
        "delivery",
        "price",
        "staff",
        "packaging",
        "app",
    }
    assert result.urgency in {"low", "medium", "high"}


def test_analysis_contains_all_six_required_fields():
    result = valid_analysis()

    expected_fields = {
        "sentiment",
        "category",
        "urgency",
        "language",
        "summary",
        "suggested_reply",
    }

    assert expected_fields.issubset(result.model_dump().keys())


def test_invalid_category_is_rejected():
    try:
        ReviewAnalysis(
            sentiment="negative",
            category="refund",
            urgency="high",
            language="English",
            summary="The customer requests a refund.",
            suggested_reply="Please contact our support team.",
        )
    except Exception:
        pass
    else:
        raise AssertionError("An invalid category should be rejected")


def test_successful_review_analysis_preserves_original_review():
    review = sample_review()

    result = analyze_one_review(FakeSuccessfulChain(), review)

    assert result["review_id"] == "R001"
    assert result["review_text"] == review["review_text"]
    assert result["rating"] == 1
    assert result["analysis_status"] == "completed"
    assert result["category"] == "delivery"


def test_failed_analysis_is_marked_for_manual_review():
    review = sample_review()

    result = analyze_one_review(FakeFailingChain(), review)

    assert result["review_id"] == "R001"
    assert result["analysis_status"] == "manual_review"
    assert result["sentiment"] == ""
    assert result["analysis_error"]


def test_tamil_and_tanglish_are_valid_language_labels():
    tamil = valid_analysis().model_copy(update={"language": "Tamil"})
    tanglish = valid_analysis().model_copy(update={"language": "Tanglish"})

    assert tamil.language == "Tamil"
    assert tanglish.language == "Tanglish"
