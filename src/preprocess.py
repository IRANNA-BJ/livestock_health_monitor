"""
Data Preprocessing Pipeline v2 for Livestock Health Monitoring
===============================================================

IMPROVEMENTS over v1:
- Handles new sensors: respiration_rate, temporal windowing features
- Additional engineered features for class separation
- Increased feature count (TOP_K=10) to leverage new discriminating signals
- All operations remain edge-compatible (simple arithmetic)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# v2 feature columns (includes new sensors + temporal features)
FEATURE_COLUMNS = [
    'acc_x', 'acc_y', 'acc_z', 'acc_magnitude', 'acc_variance',
    'temperature', 'temp_deviation', 'heart_rate', 'respiration_rate',
    'activity_index', 'rumination_rate',
    'hr_rolling_std', 'hr_rolling_range',
    'activity_rolling_std', 'activity_rolling_range',
    'movement_consistency', 'resp_hr_ratio'
]

TARGET_COLUMN = 'health_state'
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42
TOP_K_FEATURES = 10  # Increased from 8 to leverage new features


def load_data(data_path):
    print(f"  Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")
    return df


def clean_data(df):
    print("\n  Cleaning data...")
    initial_count = len(df)

    df = df.drop_duplicates()
    print(f"    Removed {initial_count - len(df)} duplicates")

    missing = df[FEATURE_COLUMNS].isnull().sum().sum()
    if missing > 0:
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(method='ffill').fillna(method='bfill')
        print(f"    Filled {missing} missing values")
    else:
        print(f"    No missing values found")

    # Physiological range clamping
    df['temperature'] = df['temperature'].clip(35.0, 42.0)
    df['heart_rate'] = df['heart_rate'].clip(30, 150)
    df['respiration_rate'] = df['respiration_rate'].clip(5, 80)
    df['rumination_rate'] = df['rumination_rate'].clip(0, 80)

    df = df.dropna(subset=FEATURE_COLUMNS)
    print(f"    Final sample count: {len(df)}")
    return df


def engineer_features(df):
    """
    Create additional features. v2 adds features specifically designed
    to separate the confusing Sick/Stressed/Inactive trio.
    """
    print("\n  Engineering features...")

    # Acceleration jerk
    df['acc_jerk'] = np.sqrt(
        df['acc_x'].diff().fillna(0)**2 +
        df['acc_y'].diff().fillna(0)**2 +
        df['acc_z'].diff().fillna(0)**2
    )

    # Heart rate to temperature ratio
    df['hr_temp_ratio'] = df['heart_rate'] / df['temperature']

    # Activity-rumination interaction
    df['activity_rumination'] = df['activity_index'] * df['rumination_rate']

    # NEW: Respiration-temperature interaction
    # Sick cows have HIGH resp with HIGH temp deviation
    # Stressed cows have HIGH resp with MILD temp deviation
    df['resp_temp_interaction'] = df['respiration_rate'] * df['temp_deviation']

    # NEW: Heart rate stability index (HR_std / HR_mean)
    # Sick = unstable (high ratio), Stressed = consistently high (low ratio)
    df['hr_instability'] = df['hr_rolling_std'] / (df['heart_rate'] + 0.001)

    # NEW: Activity-movement consistency interaction
    # Inactive = low activity + high consistency
    # Sick = low activity + low consistency (tremors break consistency)
    df['activity_consistency'] = df['activity_index'] * df['movement_consistency']

    new_features = [
        'acc_jerk', 'hr_temp_ratio', 'activity_rumination',
        'resp_temp_interaction', 'hr_instability', 'activity_consistency'
    ]

    print(f"    Added {len(new_features)} engineered features:")
    for f in new_features:
        print(f"      - {f}")

    return df, FEATURE_COLUMNS + new_features


def select_features(X, y, feature_names, k=TOP_K_FEATURES):
    print(f"\n  Selecting top {k} features...")

    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    mask = selector.get_support()
    selected_names = [name for name, selected in zip(feature_names, mask) if selected]

    scores = selector.scores_
    feature_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    print(f"    Feature importance ranking:")
    for i, (name, score) in enumerate(feature_scores):
        marker = "[+]" if name in selected_names else "[-]"
        print(f"      {marker} {name}: {score:.2f}")

    print(f"    Selected features: {selected_names}")
    return X_selected, selected_names, selector


def normalize_data(X_train, X_val, X_test):
    print("\n  Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print(f"    Scaler fitted on {X_train_scaled.shape[0]} training samples")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def preprocess_pipeline(data_path=None):
    print("=" * 60)
    print("  PREPROCESSING PIPELINE v2")
    print("=" * 60)

    project_dir = os.path.dirname(os.path.dirname(__file__))
    if data_path is None:
        data_path = os.path.join(project_dir, 'data', 'livestock_sensor_data.csv')
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Step 1: Load
    print("\n[Step 1] Loading data")
    df = load_data(data_path)

    # Step 2: Clean
    print("\n[Step 2] Cleaning data")
    df = clean_data(df)

    # Step 3: Feature engineering
    print("\n[Step 3] Feature engineering")
    df, all_features = engineer_features(df)

    # Step 4: Extract
    print("\n[Step 4] Extracting features and labels")
    X = df[all_features].values
    y = df[TARGET_COLUMN].values
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Step 5: Feature selection
    print("\n[Step 5] Feature selection")
    X_selected, selected_features, selector = select_features(X, y, all_features)

    # Step 6: Split
    print("\n[Step 6] Splitting data")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_selected, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # Step 7: Normalize
    print("\n[Step 7] Normalization")
    X_train, X_val, X_test, scaler = normalize_data(X_train, X_val, X_test)

    # Step 8: Save artifacts
    print("\n[Step 8] Saving preprocessing artifacts")
    artifacts = {
        'scaler': scaler,
        'selector': selector,
        'selected_features': selected_features,
        'all_features': all_features,
        'label_map': {0: 'Healthy', 1: 'Sick', 2: 'Stressed', 3: 'Inactive'}
    }

    artifacts_path = os.path.join(models_dir, 'preprocessing_artifacts.joblib')
    joblib.dump(artifacts, artifacts_path)
    artifact_size = os.path.getsize(artifacts_path) / 1024
    print(f"  Saved to: {artifacts_path}")
    print(f"  Artifact size: {artifact_size:.1f} KB")

    print("\n" + "=" * 60)
    print("  PREPROCESSING COMPLETE [OK]")
    print("=" * 60)

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'scaler': scaler, 'selector': selector,
        'selected_features': selected_features,
        'all_features': all_features
    }


if __name__ == "__main__":
    data = preprocess_pipeline()
    print(f"\nReady for training!")
    print(f"  Training samples: {data['X_train'].shape}")
    print(f"  Selected features: {data['selected_features']}")
