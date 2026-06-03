import requests
import streamlit as st

API_URL = "https://manufacturing-defect-api.onrender.com/predict"

st.set_page_config(
    page_title="Manufacturing Defect Prediction",
    page_icon="🏭",
    layout="centered"
)

st.title("🏭 Manufacturing Defect Prediction Dashboard")

st.write(
    "This dashboard predicts potential machine failures using an XGBoost model "
    "trained on the AI4I 2020 Predictive Maintenance Dataset."
)

st.info(
    "Enter machine sensor values below and click **Predict Machine Failure** "
    "to estimate the probability of failure."
)

st.divider()

st.subheader("Machine Sensor Inputs")

air_temperature = st.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=298.1
)

process_temperature = st.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=400.0,
    value=308.6
)

rotational_speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=0,
    max_value=4000,
    value=1551
)

torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=42.8
)

tool_wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    max_value=300,
    value=0
)

machine_type = st.selectbox(
    "Machine Type",
    ["H", "L", "M"]
)

type_l = 1 if machine_type == "L" else 0
type_m = 1 if machine_type == "M" else 0

payload = {
    "Air_temperature_K": air_temperature,
    "Process_temperature_K": process_temperature,
    "Rotational_speed_rpm": rotational_speed,
    "Torque_Nm": torque,
    "Tool_wear_min": tool_wear,
    "Type_L": type_l,
    "Type_M": type_m
}

st.divider()

if st.button("Predict Machine Failure", type="primary"):
    try:
        with st.spinner("Sending data to the prediction API..."):
            response = requests.post(API_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()

            prediction = result["prediction"]
            probability = result["failure_probability"]
            interpretation = result["interpretation"]

            probability_percent = probability * 100

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("⚠️ Potential Machine Failure Predicted")
            else:
                st.success("✅ No Failure Predicted")

            st.metric(
                label="Failure Probability",
                value=f"{probability_percent:.2f}%"
            )

            st.progress(min(probability, 1.0))

            if probability < 0.20:
                st.success("🟢 Risk Level: Low")
            elif probability < 0.50:
                st.warning("🟡 Risk Level: Medium")
            else:
                st.error("🔴 Risk Level: High")

            st.write(f"**Model Interpretation:** {interpretation}")

            st.divider()

            st.subheader("Prediction Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Machine Type:** {machine_type}")
                st.write(f"**Air Temperature:** {air_temperature} K")
                st.write(f"**Process Temperature:** {process_temperature} K")

            with col2:
                st.write(f"**Rotational Speed:** {rotational_speed} rpm")
                st.write(f"**Torque:** {torque} Nm")
                st.write(f"**Tool Wear:** {tool_wear} min")

        else:
            st.error(f"API error: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error("Could not connect to the prediction API.")
        st.write(e)

st.divider()

st.caption(
    "Built by Sergio Trejo | XGBoost + FastAPI + Docker + Render + Streamlit"
)