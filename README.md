# Customer Churn Prediction App

This project predicts customer churn using a machine learning model trained on the IBM Telco Customer Churn dataset.

## Features

- Data preprocessing pipeline
- Machine learning model using Scikit-learn
- Churn probability prediction
- Batch CSV upload
- Streamlit web app
- Downloadable prediction results

## Files

- `app.py` - Streamlit application
- `requirements.txt` - Required Python libraries
- `churn_prediction_pipeline.pkl` - Trained machine learning model
- `model_metadata.json` - Model details and threshold
- `sample_input.csv` - Sample input file for testing

## How to Run Locally


```bash
pip install -r requirements.txt
streamlit run app.py
