"""
Edge Inference Engine v2 for Livestock Health Monitor
=====================================================
Updated to handle new sensors (respiration) and temporal features.
"""

import numpy as np
import time
import os
import joblib
import warnings
warnings.filterwarnings('ignore')


class LivestockHealthPredictor:
    """Lightweight inference engine for edge deployment."""

    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

        self.model = joblib.load(os.path.join(models_dir, 'best_model.joblib'))
        artifacts = joblib.load(os.path.join(models_dir, 'preprocessing_artifacts.joblib'))
        self.meta = joblib.load(os.path.join(models_dir, 'model_metadata.joblib'))

        self.scaler = artifacts['scaler']
        self.selector = artifacts['selector']
        self.all_features = artifacts['all_features']
        self.selected_features = artifacts['selected_features']
        self.label_map = artifacts['label_map']

        print(f"  Model loaded: {self.meta['name']}")
        print(f"  Features ({len(self.selected_features)}): {self.selected_features}")
        print(f"  Classes: {list(self.label_map.values())}")

    def preprocess_input(self, raw_sensor_data):
        """Preprocess raw sensor dict into model-ready input."""
        ax = raw_sensor_data['acc_x']
        ay = raw_sensor_data['acc_y']
        az = raw_sensor_data['acc_z']
        hr = raw_sensor_data['heart_rate']
        temp = raw_sensor_data['temperature']
        resp = raw_sensor_data['respiration_rate']
        rum = raw_sensor_data['rumination_rate']

        # Derived features
        raw_sensor_data['acc_magnitude'] = np.sqrt(ax**2 + ay**2 + az**2)
        raw_sensor_data['acc_variance'] = 0.01
        raw_sensor_data['temp_deviation'] = abs(temp - 38.6)
        raw_sensor_data['activity_index'] = abs(ax) + abs(ay) + abs(az - 9.81)
        raw_sensor_data['resp_hr_ratio'] = resp / (hr + 0.001)

        # Temporal features (use provided or defaults for single-sample)
        raw_sensor_data.setdefault('hr_rolling_std', raw_sensor_data.get('hr_rolling_std', 2.0))
        raw_sensor_data.setdefault('hr_rolling_range', raw_sensor_data.get('hr_rolling_range', 8.0))
        raw_sensor_data.setdefault('activity_rolling_std', raw_sensor_data.get('activity_rolling_std', 0.1))
        raw_sensor_data.setdefault('activity_rolling_range', raw_sensor_data.get('activity_rolling_range', 0.3))
        raw_sensor_data.setdefault('movement_consistency', raw_sensor_data.get('movement_consistency', 50.0))

        # Engineered features
        raw_sensor_data['acc_jerk'] = 0.0
        raw_sensor_data['hr_temp_ratio'] = hr / temp
        raw_sensor_data['activity_rumination'] = raw_sensor_data['activity_index'] * rum
        raw_sensor_data['resp_temp_interaction'] = resp * raw_sensor_data['temp_deviation']
        raw_sensor_data['hr_instability'] = raw_sensor_data['hr_rolling_std'] / (hr + 0.001)
        raw_sensor_data['activity_consistency'] = raw_sensor_data['activity_index'] * raw_sensor_data['movement_consistency']

        feature_vector = np.array([[raw_sensor_data[f] for f in self.all_features]])
        feature_selected = self.selector.transform(feature_vector)
        feature_scaled = self.scaler.transform(feature_selected)
        return feature_scaled

    def predict(self, raw_sensor_data):
        """Full inference pipeline: preprocess -> predict -> return result."""
        start = time.perf_counter()
        X = self.preprocess_input(raw_sensor_data)
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0] if hasattr(self.model, 'predict_proba') else None
        latency = (time.perf_counter() - start) * 1000

        return {
            'health_state': int(prediction),
            'health_label': self.label_map[prediction],
            'confidence': float(max(proba)) if proba is not None else None,
            'probabilities': {self.label_map[i]: float(p) for i, p in enumerate(proba)} if proba is not None else None,
            'latency_ms': latency
        }


def run_demo():
    """Run demonstration with realistic sensor inputs including temporal context."""
    print("=" * 60)
    print("  LIVESTOCK HEALTH MONITOR v2 - LIVE DEMO")
    print("  (with respiration sensor + temporal features)")
    print("=" * 60)

    predictor = LivestockHealthPredictor()

    # Test cases now include respiration_rate AND temporal context features
    # that capture what a 30-second observation window would produce
    test_cases = [
        {
            'name': '[COW] Healthy Grazing Cow',
            'data': {
                'acc_x': 0.18, 'acc_y': 0.12, 'acc_z': 9.83,
                'temperature': 38.5, 'heart_rate': 64.0,
                'respiration_rate': 23.0,
                'rumination_rate': 58.0,
                'hr_rolling_std': 6.5, 'hr_rolling_range': 18.0,
                'activity_rolling_std': 0.25, 'activity_rolling_range': 0.9,
                'movement_consistency': 40.0
            }
        },
        {
            'name': '[SICK] Sick Cow (Fever + Unstable HR)',
            'data': {
                'acc_x': 0.03, 'acc_y': 0.02, 'acc_z': 9.80,
                'temperature': 40.5, 'heart_rate': 92.0,
                'respiration_rate': 45.0,
                'rumination_rate': 15.0,
                # KEY Sick signatures from actual data distribution:
                'hr_rolling_std': 18.0, 'hr_rolling_range': 74.0,  # Highly unstable HR
                'activity_rolling_std': 0.28, 'activity_rolling_range': 1.2,  # Tremor spikes
                'movement_consistency': 200.0  # Still lying down (high consistency, low activity)
            }
        },
        {
            'name': '[STRESS] Stressed Cow (Heat Stress)',
            'data': {
                'acc_x': 0.40, 'acc_y': 0.35, 'acc_z': 10.0,
                'temperature': 39.5, 'heart_rate': 100.0,
                'respiration_rate': 42.0,
                'rumination_rate': 30.0,
                # KEY Stressed signatures: moderate HR variability, high activity variability
                'hr_rolling_std': 10.0, 'hr_rolling_range': 40.0,
                'activity_rolling_std': 0.70, 'activity_rolling_range': 2.8,
                'movement_consistency': 22.0  # Agitated, inconsistent
            }
        },
        {
            'name': '[REST] Inactive/Resting Cow',
            'data': {
                'acc_x': 0.01, 'acc_y': 0.005, 'acc_z': 9.81,
                'temperature': 37.9, 'heart_rate': 48.0,
                'respiration_rate': 14.0,
                'rumination_rate': 22.0,
                # KEY Inactive signatures: very stable HR, near-zero activity variation
                'hr_rolling_std': 4.0, 'hr_rolling_range': 16.0,
                'activity_rolling_std': 0.017, 'activity_rolling_range': 0.07,
                'movement_consistency': 900.0  # Ultra-consistent (still body)
            }
        },
        {
            'name': '[COW] Healthy Walking Cow',
            'data': {
                'acc_x': 0.25, 'acc_y': 0.18, 'acc_z': 9.88,
                'temperature': 38.7, 'heart_rate': 70.0,
                'respiration_rate': 26.0,
                'rumination_rate': 52.0,
                'hr_rolling_std': 7.0, 'hr_rolling_range': 20.0,
                'activity_rolling_std': 0.28, 'activity_rolling_range': 1.0,
                'movement_consistency': 35.0
            }
        },
        {
            'name': '[SICK] Sick Cow (Hypothermia + Depressed)',
            'data': {
                'acc_x': 0.03, 'acc_y': 0.02, 'acc_z': 9.79,
                'temperature': 37.0, 'heart_rate': 88.0,
                'respiration_rate': 10.0,
                'rumination_rate': 10.0,
                # Hypothermic sick: unstable HR, lying still
                'hr_rolling_std': 19.0, 'hr_rolling_range': 80.0,
                'activity_rolling_std': 0.20, 'activity_rolling_range': 1.0,
                'movement_consistency': 190.0
            }
        },
    ]

    print(f"\n  Running {len(test_cases)} test scenarios...\n")

    for i, tc in enumerate(test_cases, 1):
        print(f"  {'-'*55}")
        print(f"  Test {i}: {tc['name']}")
        print(f"  {'-'*55}")

        data = tc['data']
        print(f"    Input Sensors:")
        print(f"      Accelerometer: X={data['acc_x']:.2f} Y={data['acc_y']:.2f} Z={data['acc_z']:.2f} g")
        print(f"      Temperature:   {data['temperature']:.1f} deg C")
        print(f"      Heart Rate:    {data['heart_rate']:.0f} BPM")
        print(f"      Respiration:   {data['respiration_rate']:.0f} breaths/min")
        print(f"      Rumination:    {data['rumination_rate']:.0f} chews/min")
        print(f"      Move Consist.: {data['movement_consistency']:.1f}")

        result = predictor.predict(data)

        icon = {'Healthy': '[GREEN]', 'Sick': '[RED]', 'Stressed': '[YELLOW]', 'Inactive': '[BLUE]'}
        state_icon = icon.get(result['health_label'], '[?]')

        print(f"")
        print(f"    +--- PREDICTION -----------------------+")
        print(f"    |  State: {state_icon} {result['health_label']:<20s}     |")
        if result['confidence']:
            print(f"    |  Confidence: {result['confidence']*100:.1f}%                  |")
        print(f"    |  Latency: {result['latency_ms']:.3f} ms               |")
        print(f"    +-----------------------------------------+")

        if result['probabilities']:
            print(f"    Probability Distribution:")
            for label, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
                bar = '#' * int(prob * 30)
                print(f"      {label:<10s}: {prob:.3f} {bar}")
        print()

    # Throughput test
    print(f"\n  {'='*55}")
    print(f"  THROUGHPUT BENCHMARK")
    print(f"  {'='*55}")
    sample = test_cases[0]['data']
    n_inferences = 10000
    start = time.perf_counter()
    for _ in range(n_inferences):
        predictor.predict(sample)
    total_time = time.perf_counter() - start
    throughput = n_inferences / total_time
    avg_lat = (total_time / n_inferences) * 1000

    print(f"  {n_inferences} inferences in {total_time:.2f}s")
    print(f"  Throughput: {throughput:.0f} predictions/sec")
    print(f"  Avg latency: {avg_lat:.3f} ms")
    print(f"  Meets <100ms constraint: {'PASS' if avg_lat < 100 else 'FAIL'}")

    print(f"\n  DEMO COMPLETE [OK]")


if __name__ == "__main__":
    run_demo()
