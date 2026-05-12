## Live Demo

Try the deployed app here: https://hossain-churn-prediction-app.streamlit.app

## Business Problem

Customer churn is a major issue for subscription-based businesses. This project predicts whether a customer is likely to churn based on their demographic, service usage, billing, and contract information.

## Objective

The objective is to build a machine learning model that can predict churn probability and help business teams identify high-risk customers early.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

## Machine Learning Pipeline

- Data cleaning
- Missing value handling
- Numeric and categorical preprocessing
- Model comparison
- Threshold tuning
- Model artifact saving
- Streamlit deployment

## Model Output

The app predicts:

- Churn probability
- Churn / No Churn label
- Downloadable prediction result

## Business Use Case

A retention team can use this app to prioritize customers who have a high probability of leaving and take proactive action, such as offering discounts, service upgrades, or personalized support.

## How to Run Locally


```bash
pip install -r requirements.txt
streamlit run app.py
