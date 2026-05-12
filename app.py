
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📉",
    layout="wide"
)

MODEL_PATH = "churn_prediction_pipeline.pkl"
METADATA_PATH = "model_metadata.json"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata

model, metadata = load_model()

st.title("📉 Customer Churn Prediction App")
st.write("Upload customer data and predict which customers are likely to churn.")

st.sidebar.header("Model Information")
st.sidebar.write("Model:", metadata["model_name"])
st.sidebar.write("Threshold:", round(metadata["threshold"], 2))
st.sidebar.write("ROC AUC:", round(metadata["roc_auc"], 3))

required_features = metadata["all_features"]

uploaded_file = st.file_uploader("Upload customer CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(input_df.head())

    missing_cols = [col for col in required_features if col not in input_df.columns]

    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.write("Required columns are:")
        st.write(required_features)
    else:
        input_df = input_df[required_features].copy()

        churn_probability = model.predict_proba(input_df)[:, 1]
        churn_prediction = (churn_probability >= metadata["threshold"]).astype(int)

        result_df = input_df.copy()
        result_df["churn_probability"] = churn_probability
        result_df["churn_prediction"] = churn_prediction
        result_df["churn_label"] = result_df["churn_prediction"].map({
            1: "Churn",
            0: "No Churn"
        })

        st.subheader("Prediction Results")
        st.dataframe(result_df)

        churn_count = int(result_df["churn_prediction"].sum())
        total_count = len(result_df)
        churn_rate = churn_count / total_count if total_count > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", total_count)
        col2.metric("Predicted Churn Customers", churn_count)
        col3.metric("Predicted Churn Rate", f"{churn_rate:.2%}")

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

else:
    st.info("Please upload a CSV file to generate churn predictions.")
    st.write("The uploaded file must contain these columns:")
    st.write(required_features)
