
from datetime import date

import pandas as pd

from src.services.weekly_report import calculate_weekly_metrics


def sample_reviews():
    return pd.DataFrame(
        [
            {
                "review_id": "R001",
                "date": "2026-11-15",
                "rating": 5,
                "sentiment": "positive",
                "category": "food_quality",
                "urgency": "low",
                "summary": "The food was excellent.",
                "review_text": "Great food!",
            },
            {
                "review_id": "R002",
                "date": "2026-11-16",
                "rating": 1,
                "sentiment": "negative",
                "category": "delivery",
                "urgency": "high",
                "summary": "The order arrived late.",
                "review_text": "Delivery was very late.",
            },
            {
                "review_id": "R003",
                "date": "2026-11-17",
                "rating": 3,
                "sentiment": "neutral",
                "category": "price",
                "urgency": "medium",
                "summary": "The price was acceptable.",
                "review_text": "Price is okay.",
            },
            {
                "review_id": "R004",
                "date": "2026-11-01",
                "rating": 2,
                "sentiment": "negative",
                "category": "staff",
                "urgency": "high",
                "summary": "Staff were unhelpful.",
                "review_text": "Staff were rude.",
            },
        ]
    )


def test_weekly_metrics_calculate_correctly():
    result = calculate_weekly_metrics(
        sample_reviews(),
        end_date=date(2026, 11, 17),
    )

    assert result["total_reviews"] == 3
    assert result["average_rating"] == 3.0
    assert result["sentiment_percentages"] == {
        "positive": 33.3,
        "neutral": 33.3,
        "negative": 33.3,
    }
    assert result["high_urgency_count"] == 1


def test_old_reviews_are_excluded():
    result = calculate_weekly_metrics(
        sample_reviews(),
        end_date=date(2026, 11, 17),
    )

    assert all(
        item["review_id"] != "R004"
        for item in result["evidence"]
    )


def test_top_complaints_and_praises():
    result = calculate_weekly_metrics(
        sample_reviews(),
        end_date=date(2026, 11, 17),
    )

    assert result["top_complaints"] == [
        {"category": "delivery", "count": 1}
    ]
    assert result["top_praises"] == [
        {"category": "food_quality", "count": 1}
    ]


def test_empty_dataframe_returns_zero_metrics():
    result = calculate_weekly_metrics(
        pd.DataFrame(),
        end_date=date(2026, 11, 17),
    )

    assert result["total_reviews"] == 0
    assert result["average_rating"] is None
    assert result["high_urgency_count"] == 0
    assert result["evidence"] == []


def test_missing_sentiment_and_urgency_columns():
    df = pd.DataFrame(
        [
            {
                "review_id": "R001",
                "date": "2026-11-15",
                "rating": 4,
            }
        ]
    )

    result = calculate_weekly_metrics(
        df,
        end_date=date(2026, 11, 17),
    )

    assert result["total_reviews"] == 1
    assert result["high_urgency_count"] == 0
    assert result["sentiment_percentages"]["positive"] == 0.0
