import os

import requests
import streamlit as st


st.set_page_config(page_title="Iris Classifier")
st.title("Iris Classifier")
st.write("Enter the flower measurements in centimeters to get a prediction.")

api_url = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
with st.form("prediction_form"):
    sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, value=5.1, step=0.1)
    sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, value=3.5, step=0.1)
    petal_length = st.number_input("Petal length (cm)", min_value=0.0, value=1.4, step=0.1)
    petal_width = st.number_input("Petal width (cm)", min_value=0.0, value=0.2, step=0.1)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }
    try:
        response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        st.success(f"Prediction: {result['class_name']} (class {result['prediction']})")
    except requests.exceptions.Timeout:
        st.error("The API did not respond within 10 seconds. Try again.")
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to the API at {api_url}. Start the API and try again.")
    except (requests.exceptions.RequestException, ValueError, KeyError) as exc:
        st.error(f"Prediction failed: {exc}")
