
"""Dashboard metrics and charts for Voxora AI."""

import pandas as pd
import plotly.express as px
import streamlit as st


COLORS = {
    "background": "#F5F8FC",
    "card": "#FFFFFF",
    "teal": "#078F83",
    "navy": "#14243B",
    "muted": "#66758A",
    "positive": "#16866F",
    "neutral": "#D99A24",
    "negative": "#D9534F",
}


def render_dashboard(df: pd.DataFrame) -> None:
    """Render review metrics, charts, and urgent-review queue."""

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 3rem;
            max-width: 1440px;
        }
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5EBF2;
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 3px 12px rgba(20, 36, 59, 0.04);
        }
        [data-testid="stMetricLabel"] {
            color: #66758A;
        }
        [data-testid="stMetricValue"] {
            color: #14243B;
        }
        .voxora-eyebrow {
            color: #078F83;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .voxora-subtitle {
            color: #66758A;
            margin-top: -8px;
            margin-bottom: 22px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="voxora-eyebrow">CUSTOMER INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )
    st.title("Review insights")
    st.caption("Your customer feedback at a glance.")

    if df is None or df.empty:
        st.info(
            "No reviews to display yet. Upload a review CSV to begin."
        )
        return

    data = df.copy()

    total_reviews = len(data)

    if "rating" in data.columns:
        ratings = pd.to_numeric(data["rating"], errors="coerce")
        average_rating = (
            float(ratings.mean()) if ratings.notna().any() else None
        )
    else:
        average_rating = None

    if "sentiment" in data.columns:
        sentiments = (
            data["sentiment"].astype("string").str.strip().str.lower()
        )
    else:
        sentiments = pd.Series("", index=data.index, dtype="string")

    negative_count = int(sentiments.eq("negative").sum())
    positive_count = int(sentiments.eq("positive").sum())

    if "urgency" in data.columns:
        urgency = (
            data["urgency"].astype("string").str.strip().str.lower()
        )
        high_urgency_count = int(urgency.eq("high").sum())
    else:
        high_urgency_count = 0

    metric_columns = st.columns(4)

    with metric_columns[0]:
        st.metric("Total reviews", f"{total_reviews:,}")

    with metric_columns[1]:
        st.metric(
            "Average rating",
            f"{average_rating:.2f} / 5"
            if average_rating is not None
            else "N/A",
        )

    with metric_columns[2]:
        st.metric("Negative reviews", f"{negative_count:,}")

    with metric_columns[3]:
        st.metric("High urgency", f"{high_urgency_count:,}")

    st.divider()

    left_chart, right_chart = st.columns(2)

    with left_chart:
        st.subheader("Sentiment distribution")

        sentiment_counts = (
            sentiments[sentiments.isin(["positive", "neutral", "negative"])]
            .value_counts()
            .reindex(["positive", "neutral", "negative"], fill_value=0)
            .rename_axis("Sentiment")
            .reset_index(name="Reviews")
        )

        if int(sentiment_counts["Reviews"].sum()) > 0:
            fig = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Reviews",
                hole=0.68,
                color="Sentiment",
                color_discrete_map={
                    "positive": COLORS["positive"],
                    "neutral": COLORS["neutral"],
                    "negative": COLORS["negative"],
                },
            )
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate="%{label}: %{value} reviews<extra></extra>",
            )
            fig.update_layout(
                margin=dict(t=20, b=20, l=10, r=10),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                font_color=COLORS["navy"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sentiment charts will appear after AI analysis.")

    with right_chart:
        st.subheader("Rating distribution")

        if "rating" in data.columns:
            rating_data = pd.to_numeric(
                data["rating"], errors="coerce"
            ).dropna()

            if not rating_data.empty:
                rating_counts = (
                    rating_data.astype(int)
                    .value_counts()
                    .reindex([1, 2, 3, 4, 5], fill_value=0)
                    .rename_axis("Rating")
                    .reset_index(name="Reviews")
                )

                fig = px.bar(
                    rating_counts,
                    x="Rating",
                    y="Reviews",
                    text="Reviews",
                    color="Rating",
                    color_continuous_scale=["#D9534F", "#D99A24", "#078F83"],
                )
                fig.update_layout(
                    margin=dict(t=20, b=20, l=10, r=10),
                    coloraxis_showscale=False,
                    xaxis=dict(dtick=1),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=COLORS["navy"],
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No valid ratings are available.")
        else:
            st.info("Rating data is not available.")

    st.divider()

    if "category" in data.columns:
        st.subheader("Customer feedback categories")

        category_data = data.copy()
        category_data["category"] = (
            category_data["category"].astype("string").str.strip()
        )
        category_data = category_data[
            category_data["category"].notna()
            & category_data["category"].ne("")
        ]

        if not category_data.empty:
            category_counts = (
                category_data["category"]
                .value_counts()
                .rename_axis("Category")
                .reset_index(name="Reviews")
            )

            fig = px.bar(
                category_counts,
                x="Reviews",
                y="Category",
                orientation="h",
                text="Reviews",
                color_discrete_sequence=[COLORS["teal"]],
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                margin=dict(t=15, b=15, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=COLORS["navy"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Categories will appear after AI analysis.")

    st.divider()
    st.subheader("Urgent review queue")

    if "urgency" in data.columns:
        urgent_mask = (
            data["urgency"].astype("string").str.strip().str.lower().eq("high")
        )
        urgent_reviews = data.loc[urgent_mask].copy()

        display_columns = [
            column
            for column in (
                "review_id",
                "date",
                "rating",
                "category",
                "summary",
                "review_text",
                "analysis_status",
            )
            if column in urgent_reviews.columns
        ]

        if urgent_reviews.empty:
            st.success("No high-urgency reviews are currently flagged.")
        else:
            st.warning(
                f"{len(urgent_reviews)} review(s) need priority attention."
            )
            st.dataframe(
                urgent_reviews[display_columns],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Urgency information will appear after AI analysis.")

    st.caption(
        f"Showing {total_reviews:,} review(s). "
        f"{positive_count:,} positive and {negative_count:,} negative."
    )
