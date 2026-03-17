import streamlit as st
import requests

API_HOST = "api"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}/predict"

st.title("ML Factory — Test Iris Model")

sepal_length = st.number_input("Sepal length", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal width", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal length", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal width", 0.0, 10.0, 0.2)

if st.button("Predict"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
        if "prediction" in result:
            st.success(f"Predicted class: {result['prediction']} (model version {result['model_version']})")
        else:
            st.error(result.get("error", "Unknown error"))
    except Exception as e:
        st.error(f"Failed to call API: {e}")