# 📈 Sales Forecasting & Demand Intelligence System

<div align="center">

### Machine Learning · Time Series · Data Science · Business Intelligence

**End-to-End Sales Forecasting & Demand Intelligence**

<br>

`Python` · `XGBoost` · `SARIMA` · `Prophet` · `Scikit-learn` · `Streamlit`

<br>

> **Turning historical sales data into forecasts, patterns, and actionable business insights.**

</div>

---

## About

Retail forecasting is not simply about predicting how much a company will sell next month.

It is about understanding demand, identifying unusual patterns, and making better decisions before those patterns become business problems.

Overstocking increases storage costs. Understocking leads to missed sales opportunities. Poor demand estimates make inventory planning unnecessarily difficult.

This project explores how machine learning, statistical forecasting, anomaly detection, and clustering can work together to address these challenges.

The goal was to build an **end-to-end Sales Forecasting and Demand Intelligence System** that moves from raw sales data to predictive insights and an interactive business dashboard.

Rather than relying on a single forecasting algorithm, I implemented and compared multiple approaches to understand their performance and practical trade-offs.

---

## Project overview

```text
historical sales data
        ↓
data cleaning & preprocessing
        ↓
exploratory data analysis
        ↓
time series analysis
        ↓
feature engineering
        ↓
forecasting models
        ↓
model evaluation
        ↓
anomaly detection
        ↓
product demand segmentation
        ↓
interactive Streamlit dashboard
```

The system combines three major analytical components:

- **Forecasting** — predicting future sales using statistical and machine learning models.
- **Anomaly Detection** — identifying unusual sales spikes and drops.
- **Demand Segmentation** — grouping products based on their demand patterns.

---

# Project pipeline

These are the major stages of the project, from understanding the dataset to presenting the results.

<details>
<summary><strong>01 · Data Loading, Cleaning & Feature Engineering</strong></summary>

The first stage focused on preparing the datasets for analysis and modelling.

**Primary dataset**

`Superstore Sales Dataset (train.csv)`

- Approximately four years of historical sales data.
- Used as the primary dataset for sales forecasting and business analysis.

**Secondary dataset**

`Video Game Sales Dataset (vgsales.csv)`

- Used for additional multi-source analysis practice.

**Work performed**

- Data loading and inspection using Pandas.
- Data cleaning and preprocessing.
- Handling missing values and checking data consistency.
- Exploratory Data Analysis (EDA).
- Feature engineering for forecasting.
- Understanding sales distributions and trends.

The objective was to understand the data before selecting forecasting models.

</details>

<details>
<summary><strong>02 · Time Series Analysis</strong></summary>

Before forecasting, I explored the underlying behaviour of historical sales.

**Analysis performed**

- Trend analysis
- Seasonality analysis
- Residual analysis
- Augmented Dickey-Fuller (ADF) Test
- Differencing

The focus was on understanding whether sales followed consistent temporal patterns and examining stationarity before applying statistical forecasting techniques.

</details>

<details>
<summary><strong>03 · Sales Forecasting & Model Comparison</strong></summary>

Implemented three different forecasting approaches to compare statistical modelling with machine learning.

**Models**

| Model | Approach |
|---|---|
| SARIMA | Statistical time series forecasting |
| Facebook Prophet | Trend and seasonality-based forecasting |
| XGBoost | Gradient boosting regression |

Models were evaluated using:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- MAPE — Mean Absolute Percentage Error

**Model performance**

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| SARIMA | 18,500.00 | 21,000.00 | 18.50% |
| Prophet | 20,250.79 | 22,318.41 | 21.86% |
| XGBoost | 13,915.32 | 18,893.85 | 13.29% |

**What the comparison showed**

XGBoost achieved the lowest error across all three evaluation metrics.

This made it the best-performing model among the three tested approaches and the selected model for the dashboard's production forecasting workflow.

</details>

<details>
<summary><strong>04 · Category & Region-Level Forecasting</strong></summary>

Sales forecasting was extended beyond the overall dataset to examine different business segments.

**Product categories**

- Furniture
- Office Supplies
- Technology

**Geographical regions**

- East
- West

The objective was to explore how forecasting results differ across product categories and geographical regions.

This provides a more granular view of sales behaviour instead of relying exclusively on overall sales predictions.

</details>

<details>
<summary><strong>05 · Anomaly Detection</strong></summary>

Not every unusual sales movement represents normal demand behaviour.

This stage focused on identifying unusual sales spikes and drops using two anomaly detection approaches.

**Methods**

`Isolation Forest` · `Z-Score Detection`

The analysis helps identify:

- Unusual sales increases.
- Unexpected sales decreases.
- Potentially abnormal sales observations.
- Patterns that may require further investigation.

The goal was to complement forecasting with a way to identify deviations from historical sales behaviour.

</details>

<details>
<summary><strong>06 · Product Demand Segmentation</strong></summary>

Products do not always follow the same demand patterns.

To explore these differences, I implemented K-Means clustering to group products based on demand-related patterns.

**Methods**

`K-Means Clustering` · `PCA Visualization`

The workflow included:

- Preparing product-level features.
- Applying K-Means clustering.
- Grouping products into demand segments.
- Using PCA for dimensionality reduction and visualization.

This adds another analytical layer to the project: understanding how products behave rather than simply predicting their future sales.

</details>

<details>
<summary><strong>07 · Interactive Business Dashboard</strong></summary>

The final stage brings the analytical components together through an interactive Streamlit dashboard.

**Dashboard modules**

| Module | Purpose |
|---|---|
| Sales Overview | Overall sales performance and KPIs |
| Forecast Explorer | Explore predicted future sales |
| Anomaly Report | Identify unusual sales patterns |
| Product Demand Segments | Explore product clusters |
| Model Performance | Compare forecasting models |

**Interface features**

- KPI cards
- Interactive filters
- Sales trend visualizations
- Forecast exploration
- Product cluster visualization
- Anomaly detection reports
- Forecasting model comparison

The dashboard provides a single interface for exploring historical sales, future predictions, and demand-related insights.

</details>

---

# Results & observations

The forecasting comparison produced the following results:

| Evaluation metric | Best-performing model |
|---|---|
| MAE | XGBoost |
| RMSE | XGBoost |
| MAPE | XGBoost |

<badge color="success">Selected model · XGBoost</badge>

<text color="secondary" size="xs">Best results among the three evaluated forecasting models.</text>

XGBoost achieved:

- MAE: `13,915.32`
- RMSE: `18,893.85`
- MAPE: `13.29%`

Compared with SARIMA and Prophet, XGBoost produced lower forecasting errors across all evaluated metrics.

One of the important takeaways was that comparing multiple forecasting approaches provides a more meaningful basis for model selection than choosing an algorithm based on popularity alone.

---

# Dashboard

The dashboard is designed around five analytical views.

**01 — Sales Overview**

Overall business performance, KPI cards, and historical sales trends.

**02 — Forecast Explorer**

Interactive exploration of future sales predictions.

**03 — Anomaly Report**

Identification and visualization of unusual sales behaviour.

**04 — Product Demand Segments**

Exploration of product groups generated through K-Means clustering.

**05 — Model Performance**

Comparison of SARIMA, Prophet, and XGBoost using forecasting evaluation metrics.

---

# Toolbox

### Programming

`Python`

### Data Processing & Analysis

`Pandas` · `NumPy`

### Data Visualization

`Matplotlib` · `Plotly`

### Statistical Analysis & Forecasting

`Statsmodels` · `Prophet` · `SARIMA`

### Machine Learning

`XGBoost` · `Scikit-learn` · `K-Means` · `Isolation Forest` · `PCA`

### Dashboard & Development

`Streamlit` · `Git` · `GitHub`

---

# Repository structure

```text
SalesForecasting/
│
├── analysis.ipynb
│
├── app.py
│
├── requirements.txt
│
├── train.csv
│
├── vgsales.csv
│
├── charts/
│
└── README.md
```

---

# Run locally

### 01 · Clone the repository

```bash
git clone https://github.com/aimlwithbobby/SalesForecasting.git
```

### 02 · Navigate to the project

```bash
cd SalesForecasting
```

### 03 · Create a virtual environment

```bash
python -m venv .venv
```

### 04 · Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 05 · Install dependencies

```bash
pip install -r requirements.txt
```

### 06 · Launch the dashboard

```bash
streamlit run app.py
```

---

# Future directions

There is still room to extend the system beyond its current implementation.

Some areas worth exploring:

- **Deep Learning:** Experimenting with LSTM-based time series forecasting.
- **Real-Time Forecasting:** Integrating continuously updated sales data.
- **Database Integration:** Connecting the application to a live database.
- **Cloud Deployment:** Making the dashboard accessible through cloud infrastructure.
- **Automated Retraining:** Developing a workflow for periodic model retraining.

The next step would be moving from an offline analytical workflow toward a continuously updated demand intelligence system.

---

# What this project explores

Building this project provided an opportunity to work across several areas of applied data science:

`data preprocessing` · `exploratory analysis` · `time series` · `statistical modelling` · `machine learning` · `anomaly detection` · `clustering` · `data visualization` · `dashboard development`

More importantly, it reinforced an idea I find central to applied machine learning:

> **A useful forecasting system is not just about predicting the future. It is about making the prediction understandable and useful for a business decision.**

---

<div align="center">

### Built with Python, Machine Learning & Data Science.

`Forecasting` · `Demand Intelligence` · `Business Analytics`

<br>

**Bobby**

AI/ML Engineering Enthusiast

<div align="center">

[GitHub](https://github.com/aimlwithbobby) · [Portfolio](https://bobbybalyan.vercel.app) · [LinkedIn](https://www.linkedin.com/in/bobbybalyan)

<br>

</div>

<br>

<sub>Machine Learning · Time Series Analysis · Data Science</sub>

</div>