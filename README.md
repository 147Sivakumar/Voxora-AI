# Voxora AI
### Every voice. Clearer insights.

Voxora AI is an AI-powered customer review analytics application that transforms customer feedback into meaningful insights. It helps businesses understand customer sentiment, identify recurring complaints, discover positive feedback, and make data-driven decisions.

## ✨ Features

- **Review Analysis** — Upload customer reviews using CSV files.
- **Sentiment Analysis** — Understand positive, neutral, and negative customer feedback.
- **Rating Insights** — Track review counts and average customer ratings.
- **Weekly Reports** — Generate summaries of customer feedback and rating trends.
- **Complaint Detection** — Identify recurring issues and areas that need attention.
- **Positive Feedback Insights** — Discover what customers appreciate most.
- **Recommended Actions** — Generate actionable suggestions based on review analysis.
- **Review Explorer** — Explore and investigate individual customer reviews.
- **CSV Export** — Export supporting review data for further analysis.
- **Data Validation** — Validate uploaded CSV files before analysis.

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- OpenAI API
- Pytest

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace `YOUR_USERNAME` and `YOUR_REPOSITORY` with your GitHub details.

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows Command Prompt:**

```cmd
venv\Scripts\activate
```

### 4. Install dependencies

If the project contains a `requirements.txt` file, run:

```cmd
pip install -r requirements.txt
```

### 5. Configure your API key

Create a `.env` file in the project root and add:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Replace the placeholder with your own OpenAI API key. Never commit your `.env` file or expose your API key publicly.

### 6. Run the application

```cmd
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

## 📊 How It Works

1. Upload a CSV file containing customer reviews.
2. Validate the review data.
3. Analyze customer sentiment and ratings.
4. Explore review insights and recurring customer concerns.
5. Generate weekly reports and recommended actions.
6. Export supporting data when needed.

## 🔐 Security

- Keep API keys and credentials private.
- Exclude `.env` and your virtual environment from Git.
- Use `.env.example` with placeholder values when sharing configuration instructions.
- Review files for sensitive information before publishing the repository.

## 🧪 Testing

Run the project tests with:

```cmd
python -m pytest -v tests
```

## 🎯 Project Goal

Voxora AI aims to help businesses listen to their customers, understand their experiences, and turn feedback into practical improvements.

## 📌 Project Status

Actively developed.

## 📄 License

No license has been specified yet. Add a license before allowing others to reuse, modify, or distribute the project under defined terms.

---

**Voxora AI — Every voice. Clearer insights.**
