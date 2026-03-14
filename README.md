 💹 Revenue Target Feasibility Analyzer

A self-serve data analytics web app that helps sales managers, business analysts, 
and small business owners evaluate whether their revenue target is realistic — 
based on historical sales data and Linear Regression forecasting.

 🎯 The Problem It Solves

Every quarter, businesses set revenue targets. But most teams have no quick way 
to check if those targets are actually achievable based on their own historical 
data — without a data team or expensive BI tools.

This tool gives you a verdict in under 10 seconds:
Upload a CSV → Get a forecast → Know if your target is realistic.
 ✨ Features

- 📊 Business KPIs — Total Sales, Profit, Orders, Margin, Avg Order Value
- 🤖 Revenue Forecasting — Linear Regression with up to 24-month forecast
- 🎯 Feasibility Verdict — Achievable / Challenging / Unrealistic with score
- 📈 Interactive Chart — Hover to see exact values (powered by Plotly)
- 🔎 EDA Dashboard — Monthly trends, Category & Region analysis, 
                          Top Products, Correlation Heatmap
- 📥 PDF Report — Download a full analysis report with one click
- 🔒 100% Local — Your data never leaves your machine

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Forecasting Model | Scikit-learn Linear Regression |
| Charts | Matplotlib, Seaborn, Plotly |
| PDF Export | fpdf2 |
| Language | Python 3 |

## 🚀 Getting Started

1. Clone the repository
   git clone https://github.com/yourusername/revenue-target-feasibility-analyzer.git
   cd revenue-target-feasibility-analyzer

2. Install dependencies
   pip install -r requirements.txt

3. Run the app
   streamlit run app.py

4. Open your browser at http://localhost:8501

## 📂 Expected Dataset Format

Upload any sales CSV with these columns:

| Column | Description |
|---|---|
| Order Date | Date of the order |
| Sales | Revenue from the order |
| Profit | Profit from the order |
| Order ID | Unique order identifier |
| Category | Product category |
| Region | Sales region |

Recommended dataset: Superstore Sales Dataset (available on Kaggle)


