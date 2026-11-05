"""Unsupervised anomaly model; fit only on synthetic normal training data."""
from sklearn.ensemble import IsolationForest
from simulator import FEATURES, generate_data


def train_model():
    training = generate_data(n=2400, scenario="Normal", seed=100)
    model = IsolationForest(n_estimators=150, contamination=0.02, random_state=42)
    model.fit(training[FEATURES])
    return model


def predict(model, df):
    result = df.copy()
    result["anomaly_score"] = -model.decision_function(result[FEATURES])
    result["ai_alert"] = (model.predict(result[FEATURES]) == -1).astype(int)
    result["rule_alert"] = ((result["speed_kmh"] > 65) |
                            (result["motor_temp_c"] > 80) |
                            (result["vibration_mm_s"] > 3.5) |
                            (result["motor_current_a"] > 70)).astype(int)
    return result
