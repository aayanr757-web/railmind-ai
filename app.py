"""RAILMIND AI: synthetic railway telemetry and anomaly monitoring demo."""
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from simulator import FEATURES, SCENARIOS, generate_data
from model import train_model, predict

st.set_page_config(page_title="RAILMIND AI", page_icon="🚆", layout="wide")
st.title("🚆 RAILMIND AI")
st.caption("Railway digital twin • anomaly detection • simulated predictive monitoring")
st.warning("Educational prototype using synthetic telemetry only. NOT a railway safety system, real-world failure predictor, or validated maintenance tool.")

@st.cache_resource
def get_model():
    return train_model()

with st.sidebar:
    st.header("Simulation controls")
    scenario = st.selectbox("Operating scenario", SCENARIOS)
    samples = st.slider("Simulation duration (seconds)", 200, 1200, 600, 100)
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42)
    st.markdown("**Model:** Isolation Forest, trained on independent normal synthetic data")
    st.caption("The simulated fault begins at 62% of the selected duration.")

model = get_model()
data = predict(model, generate_data(samples, scenario, int(seed)))
latest = data.iloc[-1]
alerts = int(data["ai_alert"].sum())
rule_alerts = int(data["rule_alert"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest speed", f"{latest['speed_kmh']:.1f} km/h")
c2.metric("Motor temperature", f"{latest['motor_temp_c']:.1f} °C")
c3.metric("AI anomaly flags", f"{alerts} / {len(data)}")
c4.metric("Rule-based flags", f"{rule_alerts} / {len(data)}")

st.subheader("Sensor telemetry")
for feature, label in [("speed_kmh", "Speed (km/h)"), ("motor_temp_c", "Motor temperature (°C)"),
                       ("vibration_mm_s", "Vibration (mm/s)"), ("motor_current_a", "Motor current (A)")]:
    with st.expander(label, expanded=feature in ("speed_kmh", "motor_temp_c")):
        st.line_chart(data.set_index("time_s")[[feature]], height=170)

st.subheader("AI anomaly score and alerts")
st.caption("Positive anomaly scores indicate samples classified as unusual by this trained model.")
st.line_chart(data.set_index("time_s")[["anomaly_score"]], height=190)
st.line_chart(data.set_index("time_s")[["ai_alert", "rule_alert", "fault_injected"]], height=170)

st.subheader("Synthetic fault detection evaluation")
if scenario == "Normal":
    st.info("No fault is injected in this scenario. Try a fault scenario to see precision, recall, and F1 against the known synthetic labels.")
else:
    truth = data["fault_injected"]
    a, b, c = st.columns(3)
    a.metric("AI precision", f"{precision_score(truth, data['ai_alert'], zero_division=0):.2f}")
    b.metric("AI recall", f"{recall_score(truth, data['ai_alert'], zero_division=0):.2f}")
    c.metric("AI F1", f"{f1_score(truth, data['ai_alert'], zero_division=0):.2f}")
    st.caption("Scores are for injected synthetic faults only; they do not establish performance on real railway equipment.")
    st.write("Confusion matrix (rows: true normal/fault; columns: predicted normal/fault)")
    st.dataframe(pd.DataFrame(confusion_matrix(truth, data["ai_alert"], labels=[0,1]),
                              index=["True normal", "True fault"],
                              columns=["Predicted normal", "Predicted fault"]))

st.subheader("Recent simulated records")
st.dataframe(data.tail(15).round(3), use_container_width=True)
st.download_button("Download simulation CSV", data.to_csv(index=False).encode("utf-8"),
                   "railmind_simulation.csv", "text/csv")
st.caption("Research demo only. AI alerts are advisory and are not connected to train control or braking.")
