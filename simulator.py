"""Deterministic synthetic locomotive telemetry for demonstration only."""
import numpy as np
import pandas as pd

FEATURES = ["speed_kmh", "motor_temp_c", "vibration_mm_s", "motor_current_a"]
SCENARIOS = ["Normal", "Motor overheating", "Excess vibration", "Overcurrent", "Overspeed", "Combined fault"]


def generate_data(n=600, scenario="Normal", seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    speed = np.clip(38 + 9 * np.sin(t / 52) + rng.normal(0, 1.4, n), 0, None)
    temp = 49 + 0.18 * speed + rng.normal(0, 1.7, n)
    vibration = 1.1 + 0.018 * speed + rng.normal(0, 0.17, n)
    current = 22 + 0.44 * speed + rng.normal(0, 2.2, n)
    fault = np.zeros(n, dtype=int)
    start = int(n * 0.62)
    if scenario != "Normal":
        fault[start:] = 1
        ramp = np.linspace(0, 1, n - start)
        if scenario in ("Motor overheating", "Combined fault"):
            temp[start:] += 26 + 12 * ramp
        if scenario in ("Excess vibration", "Combined fault"):
            vibration[start:] += 3.1 + 1.0 * ramp
        if scenario in ("Overcurrent", "Combined fault"):
            current[start:] += 32 + 8 * ramp
        if scenario in ("Overspeed", "Combined fault"):
            speed[start:] += 37 + 6 * ramp
    return pd.DataFrame({"time_s": t, "speed_kmh": speed, "motor_temp_c": temp,
                         "vibration_mm_s": np.maximum(0, vibration),
                         "motor_current_a": current, "fault_injected": fault})
