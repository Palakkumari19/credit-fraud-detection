# Credit Card Fraud Detection & Explainability

An end-to-end ML pipeline for detecting credit card fraud with 
XGBoost, SHAP explainability, and an interactive Streamlit dashboard.

🚀 **[Live Demo](your-streamlit-url-here)**

---

## Problem Statement

Credit card fraud costs the global economy $32B+ annually. 
Traditional rule-based systems generate excessive false alarms, 
frustrating customers and overwhelming fraud teams. This project 
builds a production-grade ML pipeline that:
- Detects fraud with 85% recall and 94% precision
- Explains every prediction using SHAP values
- Catches 83/98 fraud cases with only 5 false alarms

---

## Results

| Metric | Value |
|---|---|
| PR-AUC | 0.8776 |
| Recall (fraud caught) | 85% (83/98) |
| Precision | 94% |
| F1 Score | 0.89 |
| False Alarms | 5 / 56,864 |
| Optimal Threshold | 0.998 |

---

## Key Findings

### Class Imbalance
Only 0.173% of transactions are fraud (492/284,807).
A naive model predicting "all legitimate" achieves 99.8% accuracy
but catches zero fraud — demonstrating why PR-AUC and Recall
are the correct metrics here.

### Most Influential Fraud Features (SHAP)
| Feature | In Top-3 SHAP For | Direction |
|---|---|---|
| V14 | 90/98 fraud cases (92%) | Negative → Fraud |
| V7  | 65/98 fraud cases (66%) | Negative → Fraud |
| V10 | 58/98 fraud cases (59%) | Negative → Fraud |

### Threshold Tuning
Default threshold (0.5) gave F1 of 0.71.
Optimising threshold to 0.998 improved F1 to 0.89 — a 25% 
improvement with no model retraining required.

### Interesting Finding
V7 ranked 2nd in SHAP importance but had only moderate linear 
correlation in EDA — the model captures non-linear interactions
that simple correlation analysis misses.

---

## Architecture

```
Raw Data (284K transactions)
        │
        ▼
┌──────────────────┐
│   Preprocessing  │
│ RobustScaler     │
│ Train/Test Split │
│ SMOTETomek       │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Model Training  │
│ XGBoost baseline │
│ Optuna (30 trials│
│ MLflow tracking  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Threshold Tuning │
│ 0.5 → 0.998      │
│ F1: 0.71 → 0.89  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  SHAP Analysis   │
│ Global importance│
│ Per-prediction   │
│ Waterfall plots  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│ Streamlit Deploy │
│ 4-tab dashboard  │
│ Live predictor   │
└──────────────────┘
```

---

## Tech Stack

| Technology | Purpose |
|---|---|
| XGBoost | Primary classifier |
| Optuna | Hyperparameter tuning (30 trials) |
| MLflow | Experiment tracking |
| SHAP | Model explainability |
| imbalanced-learn | SMOTETomek resampling |
| Streamlit | Dashboard & deployment |
| Plotly | Interactive visualizations |

---

## Screenshots

### Dashboard Overview
[add screenshot]

### Live Fraud Detector
[add screenshot]

### SHAP Waterfall — Most Confident Fraud Case
[add screenshot]

---

## Local Setup

```bash
git clone https://github.com/Palakkumari19/credit-fraud-detection
cd credit-fraud-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download dataset from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place at `data/raw/creditcard.csv`, then run notebooks 01 and 02.

```bash
streamlit run app.py
```

---

## Author
Palak Kumari
[GitHub](https://github.com/Palakkumari19) | 
[LinkedIn](https://www.linkedin.com/in/palak-kumari-828779200/)