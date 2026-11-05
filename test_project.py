"""Quick deterministic smoke tests; run: python -m unittest -v"""
import unittest
from simulator import generate_data
from model import train_model, predict

class RailmindTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = train_model()

    def test_normal_has_no_injected_fault(self):
        self.assertEqual(int(generate_data(300)["fault_injected"].sum()), 0)

    def test_fault_injected_after_62_percent(self):
        df = generate_data(500, "Motor overheating")
        self.assertEqual(int(df["fault_injected"].sum()), 190)

    def test_model_outputs_binary_flags(self):
        df = predict(self.model, generate_data(300, "Excess vibration"))
        self.assertTrue(set(df["ai_alert"].unique()).issubset({0, 1}))
        self.assertFalse(df["anomaly_score"].isna().any())

    def test_deterministic_simulation(self):
        self.assertTrue(generate_data(300, seed=7).equals(generate_data(300, seed=7)))

if __name__ == "__main__":
    unittest.main()
