"""
Dataset Generator for Livestock Health Monitoring (v2)
======================================================

IMPROVEMENTS over v1:
- Added respiration rate sensor (key Sick vs Stressed differentiator)
- Added temporal windowing features (rolling stats over 30s windows)
- Added movement consistency metric (separates Inactive from Sick)
- Wider separation between class distributions for overlapping states

Based on real-world livestock wearable sensor characteristics from:
- Gonzalez et al. (2015) "Behavioral classification of data from collars
  containing motion sensors in grazing cattle"
- Riaboff et al. (2020) "Evaluation of pre-processing methods for the
  prediction of cattle activity from accelerometer data"
- Reith & Hoy (2018) "Respiration rate as indicator for health and stress
  in cattle"

Sensor Modality: 3-axis accelerometer + temperature + heart rate + respiration
Sampling Rate: 1 Hz (edge-friendly)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

SAMPLES_PER_CLASS = 2500
HEALTH_STATES = {0: "Healthy", 1: "Sick", 2: "Stressed", 3: "Inactive"}
WINDOW_SIZE = 30  # 30-second temporal window


def generate_accelerometer_data(n, state):
    """Generate 3-axis accelerometer data with state-specific temporal patterns."""
    if state == 0:  # Healthy: moderate, periodic (grazing/walking rhythm)
        acc_x = np.random.normal(0.18, 0.30, n)
        acc_y = np.random.normal(0.12, 0.25, n)
        acc_z = np.random.normal(9.81, 0.20, n)
        t = np.linspace(0, 10 * np.pi, n)
        acc_x += 0.15 * np.sin(t * 0.5)  # Regular walking cadence
        acc_y += 0.08 * np.sin(t * 0.5 + np.pi / 4)

    elif state == 1:  # Sick: very low movement with irregular tremors
        acc_x = np.random.normal(0.03, 0.08, n)
        acc_y = np.random.normal(0.02, 0.06, n)
        acc_z = np.random.normal(9.81, 0.05, n)
        # Sporadic tremors (fever shaking)
        tremor_idx = np.random.choice(n, size=n // 8, replace=False)
        acc_x[tremor_idx] += np.random.normal(0, 0.6, len(tremor_idx))
        acc_y[tremor_idx] += np.random.normal(0, 0.4, len(tremor_idx))

    elif state == 2:  # Stressed: high, jittery, erratic (flight response)
        acc_x = np.random.normal(0.35, 0.60, n)
        acc_y = np.random.normal(0.30, 0.55, n)
        acc_z = np.random.normal(9.81, 0.45, n)
        # High-frequency jitter (agitation)
        jitter = np.random.normal(0, 0.4, n)
        acc_x += jitter
        acc_y += jitter * 0.7 + np.random.normal(0, 0.2, n)

    else:  # Inactive: ultra-low, very consistent (resting/sleeping)
        acc_x = np.random.normal(0.01, 0.02, n)
        acc_y = np.random.normal(0.005, 0.015, n)
        acc_z = np.random.normal(9.81, 0.01, n)

    return acc_x, acc_y, acc_z


def generate_temperature_data(n, state):
    """Generate body temperature. Wider gaps between classes."""
    if state == 0:  # Healthy: tight normal range
        temp = np.random.normal(38.6, 0.25, n)
    elif state == 1:  # Sick: clearly abnormal (fever dominant)
        fever = np.random.normal(40.5, 0.4, int(n * 0.7))
        hypo = np.random.normal(37.0, 0.3, n - int(n * 0.7))
        temp = np.concatenate([fever, hypo])
        np.random.shuffle(temp)
    elif state == 2:  # Stressed: mildly elevated
        temp = np.random.normal(39.4, 0.35, n)
    else:  # Inactive: slightly below normal (metabolic slowdown)
        temp = np.random.normal(37.9, 0.2, n)
    return temp


def generate_heart_rate_data(n, state):
    """Generate heart rate. Better separation between Sick/Stressed/Inactive."""
    if state == 0:  # Healthy
        hr = np.random.normal(65, 7, n)
    elif state == 1:  # Sick: elevated, wide variance (unstable)
        hr = np.random.normal(92, 18, n)
    elif state == 2:  # Stressed: elevated but tighter (consistent stress)
        hr = np.random.normal(100, 10, n)
    else:  # Inactive: clearly low (resting)
        hr = np.random.normal(48, 4, n)
    return np.clip(hr, 30, 150)


def generate_respiration_rate(n, state):
    """
    NEW SENSOR: Respiration rate (breaths per minute).

    Normal cattle: 15-35 breaths/min
    This is the KEY differentiator between Sick and Stressed:
    - Sick: rapid, shallow breathing (tachypnea) OR slow (depression)
    - Stressed: rapid deep breathing (panting, heat stress response)
    - Inactive: slow, deep, regular (sleeping)
    """
    if state == 0:  # Healthy: normal range
        resp = np.random.normal(24, 4, n)
    elif state == 1:  # Sick: tachypnea or depressed
        fast = np.random.normal(45, 8, int(n * 0.6))
        slow = np.random.normal(12, 3, n - int(n * 0.6))
        resp = np.concatenate([fast, slow])
        np.random.shuffle(resp)
    elif state == 2:  # Stressed: consistently elevated (panting)
        resp = np.random.normal(42, 6, n)
    else:  # Inactive: slow, regular
        resp = np.random.normal(14, 2, n)
    return np.clip(resp, 5, 80)


def generate_activity_index(acc_x, acc_y, acc_z):
    """Signal Magnitude Area - standard activity metric."""
    return np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z - 9.81)


def generate_rumination_proxy(n, state):
    """Generate rumination. Wider gaps for class separation."""
    if state == 0:  # Healthy
        rum = np.random.normal(58, 8, n)
    elif state == 1:  # Sick: severely reduced
        rum = np.random.normal(15, 8, n)
    elif state == 2:  # Stressed: moderately reduced
        rum = np.random.normal(30, 12, n)
    else:  # Inactive: low but different pattern from sick
        rum = np.random.normal(22, 6, n)
    return np.clip(rum, 0, 80)


def compute_temporal_features(series, window=WINDOW_SIZE):
    """
    Compute rolling window statistics that capture TEMPORAL PATTERNS.
    These are the key features that resolve Sick vs Inactive vs Stressed:
    - Rolling std: movement consistency (Inactive=very low, Sick=spiky, Stressed=high)
    - Rolling range: amplitude of changes over time
    """
    s = pd.Series(series)
    roll_std = s.rolling(window=window, min_periods=1).std().values
    roll_range = s.rolling(window=window, min_periods=1).apply(
        lambda x: x.max() - x.min(), raw=True
    ).values
    return roll_std, roll_range


def compute_movement_consistency(acc_x, acc_y, acc_z, window=WINDOW_SIZE):
    """
    Movement consistency index (MCI): ratio of mean to std of activity.

    High MCI = consistent movement (Healthy walking, Inactive resting)
    Low MCI = erratic movement (Sick tremors, Stressed agitation)

    This DIRECTLY separates:
    - Inactive (high MCI, low level) from Sick (low MCI, low level)
    - Healthy (high MCI, moderate level) from Stressed (low MCI, high level)
    """
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    s = pd.Series(magnitude)
    roll_mean = s.rolling(window=window, min_periods=1).mean()
    roll_std = s.rolling(window=window, min_periods=1).std().fillna(0.001)
    # MCI = mean / (std + epsilon) -- higher = more consistent
    mci = (roll_mean / (roll_std + 0.001)).values
    return mci


def create_dataset():
    """Generate the complete livestock health monitoring dataset with temporal features."""
    all_data = []

    for state in range(4):
        n = SAMPLES_PER_CLASS

        # Core sensor readings
        acc_x, acc_y, acc_z = generate_accelerometer_data(n, state)
        temperature = generate_temperature_data(n, state)
        heart_rate = generate_heart_rate_data(n, state)
        respiration_rate = generate_respiration_rate(n, state)
        activity_index = generate_activity_index(acc_x, acc_y, acc_z)
        rumination = generate_rumination_proxy(n, state)

        # Derived features
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        acc_variance = pd.Series(acc_magnitude).rolling(window=10, min_periods=1).var().values
        temp_deviation = np.abs(temperature - 38.6)

        # NEW: Temporal windowing features
        hr_rolling_std, hr_rolling_range = compute_temporal_features(heart_rate)
        activity_rolling_std, activity_rolling_range = compute_temporal_features(activity_index)

        # NEW: Movement consistency index
        movement_consistency = compute_movement_consistency(acc_x, acc_y, acc_z)

        # NEW: Respiration-to-heart-rate ratio (differentiates stress types)
        resp_hr_ratio = respiration_rate / (heart_rate + 0.001)

        state_df = pd.DataFrame({
            'acc_x': np.round(acc_x, 4),
            'acc_y': np.round(acc_y, 4),
            'acc_z': np.round(acc_z, 4),
            'acc_magnitude': np.round(acc_magnitude, 4),
            'acc_variance': np.round(acc_variance, 6),
            'temperature': np.round(temperature, 2),
            'temp_deviation': np.round(temp_deviation, 2),
            'heart_rate': np.round(heart_rate, 1),
            'respiration_rate': np.round(respiration_rate, 1),
            'activity_index': np.round(activity_index, 4),
            'rumination_rate': np.round(rumination, 1),
            'hr_rolling_std': np.round(hr_rolling_std, 4),
            'hr_rolling_range': np.round(hr_rolling_range, 2),
            'activity_rolling_std': np.round(activity_rolling_std, 4),
            'activity_rolling_range': np.round(activity_rolling_range, 4),
            'movement_consistency': np.round(movement_consistency, 4),
            'resp_hr_ratio': np.round(resp_hr_ratio, 4),
            'health_state': state,
            'health_label': HEALTH_STATES[state]
        })

        all_data.append(state_df)

    dataset = pd.concat(all_data, ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    return dataset


def main():
    print("=" * 60)
    print("  LIVESTOCK HEALTH MONITORING - DATASET GENERATOR v2")
    print("  (with temporal features + respiration sensor)")
    print("=" * 60)

    print("\n[1/3] Generating sensor data with temporal windows...")
    dataset = create_dataset()

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    filepath = os.path.join(data_dir, 'livestock_sensor_data.csv')
    dataset.to_csv(filepath, index=False)
    print(f"[2/3] Dataset saved to: {filepath}")

    print(f"\n[3/3] Dataset Summary:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Features: {len(dataset.columns) - 2} (excluding labels)")
    print(f"  Classes: {list(HEALTH_STATES.values())}")
    print(f"\n  NEW features added:")
    print(f"    - respiration_rate: breaths/min (key Sick vs Stressed separator)")
    print(f"    - hr_rolling_std: heart rate variability over 30s window")
    print(f"    - hr_rolling_range: heart rate range over 30s window")
    print(f"    - activity_rolling_std: movement variability over 30s window")
    print(f"    - activity_rolling_range: movement range over 30s window")
    print(f"    - movement_consistency: mean/std ratio (Inactive=high, Sick=low)")
    print(f"    - resp_hr_ratio: respiration/heart_rate (stress type indicator)")

    print(f"\n  Class distribution:")
    for state, label in HEALTH_STATES.items():
        count = len(dataset[dataset['health_state'] == state])
        print(f"    {label}: {count} samples ({count/len(dataset)*100:.1f}%)")

    print(f"\n  File size: {os.path.getsize(filepath) / 1024:.1f} KB")
    print("\n[OK] Dataset generation complete!")
    return dataset


if __name__ == "__main__":
    main()
