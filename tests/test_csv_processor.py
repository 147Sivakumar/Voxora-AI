
import pandas as pd

from src.utils.csv_processor import validate_reviews


def sample_data():
    return pd.DataFrame(
        [
            {
                "review_id": "R001",
                "date": "2026-11-15",
                "platform": "Google",
                "rating": 5,
                "review_text": "Great food and friendly staff!",
            },
            {
                "review_id": "R002",
                "date": "2026-11-16",
                "platform": "Swiggy",
                "rating": 1,
                "review_text": "சாப்பாடு நல்லா இல்லை.",
            },
        ]
    )


def test_valid_reviews_are_accepted():
    clean, errors, warnings = validate_reviews(sample_data())

    assert not errors
    assert len(clean) == 2
    assert clean["rating"].tolist() == [5, 1]


def test_missing_required_column_is_rejected():
    df = sample_data().drop(columns=["review_text"])

    clean, errors, _ = validate_reviews(df)

    assert clean.empty
    assert any("review_text" in error for error in errors)


def test_invalid_rating_is_skipped():
    df = sample_data()
    df.loc[0, "rating"] = 7

    clean, errors, warnings = validate_reviews(df)

    assert not errors
    assert len(clean) == 1
    assert clean.iloc[0]["review_id"] == "R002"
    assert any("invalid ratings" in warning for warning in warnings)


def test_duplicate_review_id_is_skipped():
    df = sample_data()
    duplicate = df.iloc[[0]].copy()
    df = pd.concat([df, duplicate], ignore_index=True)

    clean, errors, warnings = validate_reviews(df)

    assert not errors
    assert len(clean) == 2
    assert clean["review_id"].is_unique
    assert any("duplicate review IDs" in warning for warning in warnings)


def test_empty_dataframe_is_rejected():
    clean, errors, _ = validate_reviews(pd.DataFrame())

    assert clean.empty
    assert errors


def test_tamil_text_is_preserved():
    df = sample_data()
    original = df.loc[1, "review_text"]

    clean, errors, _ = validate_reviews(df)

    assert not errors
    assert original in clean["review_text"].tolist()
