# Customer Engagement & Product Utilization Analytics for Retention Strategy

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Unified Mentor & European Central Bank (ECB) Data Science Project Submission**  
*Evaluating retail banking customer retention through behavioral interaction depth, product portfolio architecture, and predictive risk scoring.*

---

## 📌 Project Overview
Traditional retail banking models often assume that customers with high balances or stellar credit scores are naturally loyal. This research and interactive machine learning platform disprove that assumption. Analyzing a portfolio of 10,000 banking customers, this investigation reveals that **silent disengagement** and **suboptimal product bundling** represent the true leading indicators of customer attrition.

### Key Analytical Findings:
- **Product Depth Inflection:** Customers holding **2 products** exhibit the lowest churn (**7.58%**), forming the portfolio "Sweet Spot". Churn jumps to **82.71% for 3 products** and **100% for 4 products**, indicating severe dissatisfaction or misaligned bundling.
- **Silent Capital Flight:** Customers who churned held an average balance of **€91,108.54**, compared to **€72,745.30** for retained clients. Over **€185.58 Million in deposits** is at active risk.
- **Geographic Vulnerability:** German accounts churn at **32.44%**, double the attrition rate of France (16.15%) and Spain (16.67%).
- **Engagement Multiplier:** Actively engaged accounts churn at less than half the rate of dormant accounts (14.27% vs 26.85%), yielding an **Engagement Retention Ratio of 1.17x**.

---

## 🚀 Live Interactive Web Application
Experience the deployed Streamlit application on Streamlit Community Cloud:
- **Live Streamlit App:** `https://your-bank-retention-app.streamlit.app` *(Replace with your deployed URL)*
- **GitHub Repository:** `https://github.com/your-username/customer-engagement-retention`

### Core Application Capabilities:
1. **Executive Dashboard & KPIs:** Live tracking of churn rates, capital at risk, geographic disparities, and product depth indices.
2. **Real-Time Churn Risk Calculator:** Instant inference engine computing churn probability (0–100%) with automated triage flags (High, Moderate, Stable).
3. **What-If Scenario Simulator:** Dynamic experimentation enabling relationship managers to observe risk reduction when turning inactive clients active or optimizing product portfolios.
4. **Feature Importance & Drivers:** Complete transparency into model mechanics via tree-based relative importance weights.
5. **High-Value Disengaged Customer Screener:** Immediate filtering of high-net-worth accounts exhibiting dormancy markers.

---

## 📊 Machine Learning Model Benchmarks

A rigorous benchmark across classification algorithms with stratified 80/20 train-test splitting:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting Classifier (Production)** | **86.90%** | **78.66%** | **48.89%** | **60.30%** | **0.8629** |
| Random Forest Classifier | 86.60% | 80.35% | 45.21% | 57.86% | 0.8602 |
| Logistic Regression (Baseline) | 85.00% | 75.36% | 39.07% | 51.46% | 0.8386 |

---

## 🛠️ Repository File Structure

```plaintext
├── app.py                                    # Interactive Streamlit Web Application
├── model.pkl                                 # Production Serialized ML Pipeline
├── train_model.py                            # End-to-end model training & export script
├── requirements.txt                          # Production Python dependencies
├── README.md                                 # Project documentation & instructions
├── 3.csv                                     # Complete dataset (10,000 banking customers)
├── Customer_Engagement_Retention_Analysis.ipynb # Jupyter notebook with EDA & modeling
├── Research_Paper_Bank_Churn_Retention.pdf   # Full Academic Research Paper
└── Executive_Summary_Bank_Churn.pdf          # 2-Page Executive C-Suite Brief
```

---

## 💻 Local Installation & Setup

To clone and run this application locally on your machine:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/customer-engagement-retention.git
cd customer-engagement-retention

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Run the Streamlit application
streamlit run app.py
```
The application will launch automatically in your default browser at `http://localhost:8501`.

---

## ☁️ Deployment Instructions (Streamlit Community Cloud)

1. Push this repository to your personal GitHub account.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New App"** and select your repository, branch (`main`), and main file (`app.py`).
4. Click **"Deploy"**. Your application will be live in 60 seconds with an interactive URL to submit.

---

## 📄 Deliverables Summary
- **Deliverable 1:** Detailed Analytical Research Paper (`Research_Paper_Bank_Churn_Retention.pdf` / `.docx`)
- **Deliverable 2:** Live Streamlit Application (`app.py`, `model.pkl`, `requirements.txt`)
- **Deliverable 3:** Executive Summary (`Executive_Summary_Bank_Churn.pdf`)
