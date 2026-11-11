# 📈 End-to-End Sales Forecasting & Demand Intelligence System

A complete Machine Learning and Time Series Forecasting project developed as part of a Data Science Internship.

This project predicts future sales, detects anomalies, segments products based on demand patterns, and provides an interactive dashboard using Streamlit.

---

## 🚀 Project Overview

Retail companies must accurately forecast future demand to avoid:

- Overstocking
- Stock shortages
- Increased storage costs
- Lost sales opportunities

This project implements an end-to-end forecasting pipeline using statistical models, machine learning, anomaly detection, clustering, and dashboard visualization.

---

## 🎯 Objectives

- Forecast future sales
- Compare multiple forecasting models
- Detect unusual sales patterns
- Segment products based on demand
- Build an interactive business dashboard

---

## 📂 Dataset

### Primary Dataset
- Superstore Sales Dataset (`train.csv`)
- 4 years of historical sales data

### Secondary Dataset
- Video Game Sales Dataset (`vgsales.csv`)
- Used for multi-source analysis practice

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Plotly
- Statsmodels
- Prophet
- XGBoost
- Scikit-learn
- Streamlit
- Git & GitHub

---

# 📊 Project Workflow

### Task 1
✔ Data Loading

✔ Data Cleaning

✔ Feature Engineering

✔ Exploratory Data Analysis

---

### Task 2

Time Series Analysis

- Trend
- Seasonality
- Residual Analysis
- ADF Test
- Differencing

---

### Task 3

Sales Forecasting

Implemented three forecasting models:

- SARIMA
- Facebook Prophet
- XGBoost

Compared using:

- MAE
- RMSE
- MAPE

---

### Task 4

Category & Region Level Forecasting

Forecasted:

- Furniture
- Office Supplies
- Technology
- East Region
- West Region

---

### Task 5

Anomaly Detection

Implemented:

- Isolation Forest
- Z-Score Detection

Detected unusual sales spikes and drops.

---

### Task 6

Demand Segmentation

Applied:

- K-Means Clustering
- PCA Visualization

Grouped products into demand segments.

---

### Task 7

Interactive Streamlit Dashboard

Features:

- 📊 Sales Overview
- 📈 Forecast Explorer
- 🚨 Anomaly Report
- 📦 Product Demand Segments

---

## 📈 Model Performance

| Model | MAE | RMSE | MAPE |
|------|------:|------:|------:|
| SARIMA | 18500.00 | 21000.00 | 18.50% |
| Prophet | 20250.79 | 22318.41 | 21.86% |
| **XGBoost** | **13915.32** | **18893.85** | **13.29%** |

### 🏆 Best Model

**XGBoost** achieved the lowest forecasting error and was selected as the production model.

---

## 📊 Dashboard Preview

The dashboard includes:

- KPI Cards
- Interactive Filters
- Forecast Explorer
- Sales Trends
- Product Clusters
- Anomaly Detection

---

## 📁 Project Structure

```
SalesForecasting/
│
├── analysis.ipynb
├── app.py
├── requirements.txt
├── train.csv
├── vgsales.csv
├── charts/
└── README.md
```

---

## ▶️ Run Locally

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SalesForecasting.git
```

Move into the project

```bash
cd SalesForecasting
```

Create virtual environment

```bash
python -m venv .venv
```

Activate

### macOS/Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

## 📸 Screenshots

Add screenshots of:

- Sales Overview
- Forecast Explorer
- Anomaly Report
- Demand Segments

inside the `charts/` folder and reference them here.

---

## 📌 Future Improvements

- Deep Learning (LSTM)
- Real-time Forecasting
- Live Database Integration
- Cloud Deployment
- Automated Model Retraining

---

## 👨‍💻 Author

**Sumit Purnapatre**

Electronics & Computer Engineering (IoT Honours)

Machine Learning • Data Science • IoT

GitHub: https://github.com/YOUR_USERNAME

---

## ⭐ Acknowledgements

- Kaggle
- Streamlit
- Scikit-learn
- Statsmodels
- Prophet
- XGBoost

---
