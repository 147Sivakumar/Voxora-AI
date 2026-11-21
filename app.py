
"""Main Streamlit application for Voxora AI."""

import os
import re
from io import BytesIO

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.components.dashboard import render_dashboard
from src.services.review_analyzer import analyze_reviews
from src.services.weekly_report import generate_weekly_report
from src.utils.csv_processor import validate_reviews


load_dotenv()

st.set_page_config(
    page_title="Voxora AI | Review Insights",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_state():
    """Create session state values once per browser session."""
    defaults = {
        "reviews": pd.DataFrame(),
        "analysis_ready": False,
        "source_name": "",
        "weekly_report": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_results():
    """Clear the currently loaded review data and results."""
    st.session_state.reviews = pd.DataFrame()
    st.session_state.analysis_ready = False
    st.session_state.source_name = ""
    st.session_state.weekly_report = None


def dataframe_to_csv(df: pd.DataFrame) -> bytes:
    """Export review results as a UTF-8 CSV."""
    return df.to_csv(index=False).encode("utf-8-sig")



def render_sentiment_bars(metrics: dict) -> None:
    """Display sentiment percentages with distinct colors."""
    st.subheader("Sentiment breakdown")

    sentiment_settings = [
        ("positive", "Positive", "#16866F"),
        ("neutral", "Neutral", "#D99A24"),
        ("negative", "Negative", "#D9534F"),
    ]

    percentages = metrics.get("sentiment_percentages", {})

    for key, label, color in sentiment_settings:
        percentage = float(percentages.get(key, 0))
        percentage = max(0.0, min(100.0, percentage))

        st.markdown(
            f"""
            <div style="margin-bottom: 18px;">
                <div style="
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 6px;
                    font-weight: 600;
                ">
                    <span>{label}</span>
                    <span style="color: {color};">
                        {percentage:.1f}%
                    </span>
                </div>
                <div style="
                    background-color: #E5EBF2;
                    height: 12px;
                    border-radius: 20px;
                    overflow: hidden;
                ">
                    <div style="
                        background-color: {color};
                        width: {percentage}%;
                        height: 12px;
                        border-radius: 20px;
                    "></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    initialize_state()

    st.markdown(
        """
        <style>
        .stApp {
            background: #F5F8FC;
        }

        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E5EBF2;
        }

        .block-container {
            max-width: 1440px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        div.stButton > button,
        div.stDownloadButton > button {
            border-radius: 10px;
            font-weight: 600;
        }

        /* Hide Streamlit's built-in top header */
        header[data-testid="stHeader"] {
               display: none;
        }

        /* Remove the empty space above the app content */
        div[data-testid="stAppViewBlockContainer"] {
             padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Workspace")
        st.divider()

        page = st.radio(
            "Workspace",
            [
                "Dashboard",
                "Analyze reviews",
                "Weekly report",
                "Review explorer",
            ],
        )

        st.divider()
        st.markdown("**Your review workspace**")
        st.caption(
            "Upload customer feedback, analyze its meaning, "
            "and identify areas for improvement."
        )

        if st.button("Clear current results", use_container_width=True):
            clear_results()
            st.rerun()

        st.divider()
        st.caption("Voxora AI · Customer review intelligence")

    st.markdown(
    """
    <div style="
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 14px 20px;
        margin-bottom: 24px;
        background: white;
        border: 1px solid #E5EBF2;
        border-radius: 14px;
    ">
        <div style="
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #E1F5F1;
            border-radius: 12px;
            color: #078F83;
            font-size: 25px;
        ">◈</div>
        <div>
            <div style="
                color: #14243B;
                font-size: 22px;
                font-weight: 750;
                line-height: 1.3;
            ">Voxora AI</div>
            <div style="
                color: #64748B;
                font-size: 13px;
            ">Every voice. Clearer insights.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

    # ---------------------------------------------------------
    # Dashboard
    # ---------------------------------------------------------
    if page == "Dashboard":
        render_dashboard(st.session_state.reviews)

    # ---------------------------------------------------------
    # Analyze reviews
    # ---------------------------------------------------------
    elif page == "Analyze reviews":
        st.subheader("Upload customer reviews")
        st.write(
            "Upload a CSV containing review IDs, dates, platforms, "
            "ratings, and customer review text."
        )

        uploaded_file = st.file_uploader(
            "Choose a review CSV file",
            type=["csv"],
            help=(
                "Required columns: review_id, date, platform, "
                "rating, review_text"
            ),
        )

        if uploaded_file is not None:
            try:
                raw_df = pd.read_csv(
                    BytesIO(uploaded_file.getvalue()),
                    encoding="utf-8-sig",
                )
            except (
                UnicodeDecodeError,
                pd.errors.ParserError,
                ValueError,
            ):
                st.error(
                    "The CSV could not be read. Please check its encoding "
                    "and formatting, then try again."
                )
                raw_df = None

            if raw_df is not None:
                clean_df, errors, warnings = validate_reviews(raw_df)

                for error in errors:
                    st.error(error)

                for warning in warnings:
                    st.warning(warning)

                if not clean_df.empty:
                    st.success(
                        f"{len(clean_df)} valid review(s) ready for analysis."
                    )

                    st.dataframe(
                        clean_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    if st.button(
                        "Analyze reviews with AI",
                        type="primary",
                        use_container_width=True,
                    ):
                        if not os.getenv("OPENAI_API_KEY"):
                            st.error(
                                "An OpenAI API key is required for AI "
                                "analysis. Add it to your local .env file "
                                "and restart the app."
                            )
                        else:
                            try:
                                progress = st.progress(0)
                                status = st.empty()

                                status.info(
                                    "Analyzing reviews. Please wait..."
                                )
                                progress.progress(25)

                                review_records = clean_df.to_dict(
                                    orient="records"
                                )

                                results = analyze_reviews(
                                    review_records,
                                    max_concurrency=5,
                                )

                                progress.progress(100)
                                status.empty()

                                st.session_state.reviews = pd.DataFrame(
                                    results
                                )
                                st.session_state.analysis_ready = True
                                st.session_state.source_name = (
                                    uploaded_file.name
                                )
                                st.session_state.weekly_report = None

                                completed = sum(
                                    row.get("analysis_status") == "completed"
                                    for row in results
                                )
                                manual = len(results) - completed

                                st.success(
                                    f"Analysis finished: {completed} "
                                    f"completed, {manual} needing manual review."
                                )
                                st.rerun()

                            except Exception:
                                st.error(
                                    "Analysis could not be completed. "
                                    "Check your API configuration and "
                                    "try again."
                                )

    # ---------------------------------------------------------
    # Weekly report
    # ---------------------------------------------------------
    elif page == "Weekly report":
        st.subheader("Weekly review report")

        if st.session_state.reviews.empty:
            st.info(
                "Analyze a CSV first. Weekly insights will appear here "
                "after review analysis."
            )
        else:
            if st.button(
                "Generate weekly report",
                type="primary",
                use_container_width=True,
            ):
                if not os.getenv("OPENAI_API_KEY"):
                    st.error(
                        "An OpenAI API key is required to generate "
                        "the written weekly report."
                    )
                else:
                    try:
                        with st.spinner("Preparing weekly insights..."):
                            report = generate_weekly_report(
                                st.session_state.reviews
                            )

                        st.session_state.weekly_report = report

                    except Exception:
                        st.error(
                            "The weekly report could not be generated. "
                            "Please check your API configuration and retry."
                        )

            report = st.session_state.weekly_report

            if report:
                metrics = report["metrics"]

                st.markdown(
                    f"**Reporting period:** {metrics['period_start']} "
                    f"to {metrics['period_end']}"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Reviews",
                    metrics["total_reviews"],
                )

                average_rating = metrics["average_rating"]

                col2.metric(
                    "Average rating",
                    (
                        f"{average_rating:.2f} / 5"
                        if average_rating is not None
                        else "N/A"
                    ),
                )

                col3.metric(
                    "High urgency",
                    metrics["high_urgency_count"],
                )

                # Display three sentiment progress bars.
                render_sentiment_bars(metrics)

                

                st.subheader("Weekly insights")

                report_text = report.get("report", "")

                if report_text:
                    section_pattern = (
                        r"\*\*(Overview|Top Complaints|Top Praises|"
                        r"Action Items|Data Limitations):\*\*\s*"
                        r"(.*?)(?=\n\s*\*\*(?:Overview|Top Complaints|"
                        r"Top Praises|Action Items|Data Limitations):\*\*|\Z)"
                    )

                    sections = {
                        match.group(1): match.group(2).strip()
                        for match in re.finditer(
                            section_pattern,
                            report_text,
                            re.DOTALL,
                        )
                    }

                    overview = sections.get("Overview")

                    if overview:
                        st.markdown("#### Key takeaway")
                        st.markdown(overview)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("#### Top complaints")
                        complaints = sections.get("Top Complaints")

                        if complaints:
                            with st.container(border=True):
                                st.markdown(complaints)
                        else:
                            st.info("No complaint summary available.")

                    with col2:
                        st.markdown("#### Top praises")
                        praises = sections.get("Top Praises")

                        if praises:
                            with st.container(border=True):
                                st.markdown(praises)
                        else:
                            st.info("No praise summary available.")

                    actions = sections.get("Action Items")

                    if actions:
                        with st.expander(
                            "Recommended actions",
                            expanded=True,
                        ):
                            st.markdown(actions)

                    limitations = sections.get("Data Limitations")

                    if limitations:
                        with st.expander("Data limitations"):
                            st.markdown(limitations)

                    if not sections:
                        st.info(
                            "The report could not be separated "
                            "into sections. Showing the original report."
                        )
                        st.markdown(report_text)

                else:
                    st.info("No weekly insights are available yet.")

                with st.expander("View supporting review evidence"):
                    evidence = metrics.get("evidence", [])

                    if evidence:
                        st.dataframe(
                            pd.DataFrame(evidence),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info(
                            "No supporting review evidence is available."
                        )

            
                st.download_button(
                    "Download weekly report data as CSV",
                    data=dataframe_to_csv(
                        pd.DataFrame(metrics.get("evidence", []))
                    ),
                    file_name="voxora_weekly_evidence.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="weekly_evidence_download",
                )

    # ---------------------------------------------------------
    # Review explorer
    # ---------------------------------------------------------
    elif page == "Review explorer":
        st.subheader("Review explorer")

        reviews = st.session_state.reviews

        if reviews.empty:
            st.info(
                "No analyzed reviews yet. Upload and analyze a CSV first."
            )
        else:
            search_text = st.text_input(
                "Search review text, summary, or review ID"
            )

            filtered = reviews.copy()

            if search_text.strip():
                searchable_columns = [
                    column
                    for column in (
                        "review_id",
                        "review_text",
                        "summary",
                        "category",
                        "suggested_reply",
                    )
                    if column in filtered.columns
                ]

                mask = pd.Series(False, index=filtered.index)

                for column in searchable_columns:
                    mask |= (
                        filtered[column]
                        .fillna("")
                        .astype(str)
                        .str.contains(
                            search_text,
                            case=False,
                            regex=False,
                        )
                    )

                filtered = filtered.loc[mask]

            filter_columns = st.columns(3)

            if "sentiment" in filtered.columns:
                sentiments = sorted(
                    filtered["sentiment"]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                selected_sentiments = filter_columns[0].multiselect(
                    "Sentiment",
                    sentiments,
                )

                if selected_sentiments:
                    filtered = filtered[
                        filtered["sentiment"].isin(selected_sentiments)
                    ]

            if "category" in filtered.columns:
                categories = sorted(
                    filtered["category"]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                selected_categories = filter_columns[1].multiselect(
                    "Category",
                    categories,
                )

                if selected_categories:
                    filtered = filtered[
                        filtered["category"].isin(selected_categories)
                    ]

            if "urgency" in filtered.columns:
                urgencies = sorted(
                    filtered["urgency"]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                selected_urgencies = filter_columns[2].multiselect(
                    "Urgency",
                    urgencies,
                )

                if selected_urgencies:
                    filtered = filtered[
                        filtered["urgency"].isin(selected_urgencies)
                    ]

            st.caption(
                f"{len(filtered)} review(s) match your filters."
            )

            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Export filtered reviews as CSV",
                data=dataframe_to_csv(filtered),
                file_name="voxora_review_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ---------------------------------------------------------
    # Footer
    # ---------------------------------------------------------
    st.divider()

    st.caption(
        "Review insights are AI-generated and may need human verification. "
        "Verify important decisions against the original customer feedback."
    )


if __name__ == "__main__":
    main()
