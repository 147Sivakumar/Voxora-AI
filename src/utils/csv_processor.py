
"""CSV loading and validation utilities for Voxora AI."""

import pandas as pd

REQUIRED_COLUMNS = [
    "review_id",
    "date",
    "platform",
    "rating",
    "review_text",
]


def validate_reviews(df: pd.DataFrame):
    """Validate a review DataFrame.

    Returns:
        A tuple of (clean_dataframe, errors, warnings).
        Errors prevent processing; warnings identify rows that were skipped.
    """
    errors = []
    warnings = []

    if df is None or df.empty:
        return pd.DataFrame(), ["The CSV file contains no review data."], warnings

    # Normalize column names without changing the original review text.
    df = df.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]

    if df.columns.duplicated().any():
        return (
            pd.DataFrame(),
            ["The CSV contains duplicate column names."],
            warnings,
        )

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        return (
            pd.DataFrame(),
            [f"Missing required columns: {', '.join(missing)}"],
            warnings,
        )

    df = df[REQUIRED_COLUMNS].copy()

    # Remove fully empty rows.
    df = df.dropna(how="all").copy()

    if df.empty:
        return pd.DataFrame(), ["No usable review rows were found."], warnings

    # Remove duplicate review IDs, keeping the first occurrence.
    df["review_id"] = df["review_id"].astype("string").str.strip()
    duplicate_mask = (
        df["review_id"].notna()
        & df["review_id"].duplicated(keep="first")
    )
    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count:
        warnings.append(
            f"Skipped {duplicate_count} row(s) with duplicate review IDs."
        )
        df = df.loc[~duplicate_mask].copy()

    # Required text fields must contain meaningful values.
    for column in ["review_id", "platform", "review_text"]:
        df[column] = df[column].astype("string").str.strip()

    missing_text = (
        df["review_id"].isna()
        | df["review_id"].eq("")
        | df["platform"].isna()
        | df["platform"].eq("")
        | df["review_text"].isna()
        | df["review_text"].eq("")
    )

    missing_count = int(missing_text.sum())
    if missing_count:
        warnings.append(
            f"Skipped {missing_count} row(s) with missing IDs, platforms, "
            "or review text."
        )
        df = df.loc[~missing_text].copy()

    # Ratings must be numeric and between 1 and 5.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    invalid_rating = (
        df["rating"].isna()
        | ~df["rating"].between(1, 5)
    )

    invalid_count = int(invalid_rating.sum())
    if invalid_count:
        warnings.append(
            f"Skipped {invalid_count} row(s) with invalid ratings. "
            "Ratings must be between 1 and 5."
        )
        df = df.loc[~invalid_rating].copy()

    # Keep dates as parsed datetime values for reliable sorting and analysis.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    invalid_dates = int(df["date"].isna().sum())
    if invalid_dates:
        warnings.append(
            f"{invalid_dates} row(s) have invalid dates; those dates are "
            "marked as missing."
        )

    df["rating"] = df["rating"].astype(int)
    df = df.reset_index(drop=True)

    if df.empty:
        errors.append("No valid reviews remain after validation.")

    return df, errors, warnings
