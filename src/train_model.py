"""
Model Training for Livestock Health Monitoring
===============================================
Trains 3 lightweight models for edge deployment.
"""

import numpy as np
import time
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_pipeline

MODEL_CONFIGS = {
    'random_forest': {
        'model': RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5,
                                         min_samples_leaf=3, max_features='sqrt', random_state=42, n_jobs=1),
        'name': 'Random Forest', 'description': 'Balanced accuracy and size'
    },
    'logistic_regression': {
        'model': LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial',
                                     solver='lbfgs', random_state=42),
        'name': 'Logistic Regression', 'description': 'Smallest model, fastest inference'
    },
    'gradient_boosting': {
        'model': GradientBoostingClassifier(n_estimators=80, max_depth=5, learning_rate=0.1,
                                             subsample=0.8, random_state=42),
        'name': 'Gradient Boosting', 'description': 'Highest accuracy'
    }
}


def evaluate_model(model, X_test, y_test, label_map):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    names = [label_map[i] for i in sorted(label_map.keys())]
    report = classification_report(y_test, y_pred, target_names=names)
    cm = confusion_matrix(y_test, y_pred)
    return {'accuracy': acc, 'f1_score': f1, 'report': report, 'cm': cm, 'predictions': y_pred}


def measure_latency(model, X_sample, n_runs=1000):
    single = X_sample[0:1]
    for _ in range(10): model.predict(single)  # warmup
    latencies = []
    for _ in range(n_runs):
        s = time.perf_counter()
        model.predict(single)
        latencies.append((time.perf_counter() - s) * 1000)
    lat = np.array(latencies)
    return {'mean_ms': np.mean(lat), 'median_ms': np.median(lat),
            'p95_ms': np.percentile(lat, 95), 'p99_ms': np.percentile(lat, 99)}


def train_all_models():
    print("=" * 60)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 60)

    data = preprocess_pipeline()
    X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    label_map = {0: 'Healthy', 1: 'Sick', 2: 'Stressed', 3: 'Inactive'}

    project_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    all_results = {}

    for model_key, config in MODEL_CONFIGS.items():
        print(f"\n{'-'*50}")
        print(f"  MODEL: {config['name']} - {config['description']}")
        print(f"{'-'*50}")

        model = config['model']
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.2f}s")

        print(f"\n  --- Validation ---")
        val_r = evaluate_model(model, X_val, y_val, label_map)
        print(f"  Accuracy: {val_r['accuracy']:.4f} | F1: {val_r['f1_score']:.4f}")

        print(f"\n  --- Test ---")
        test_r = evaluate_model(model, X_test, y_test, label_map)
        print(f"  Accuracy: {test_r['accuracy']:.4f} | F1: {test_r['f1_score']:.4f}")
        print(f"\n{test_r['report']}")

        print(f"\n  Confusion Matrix:")
        names = [label_map[i] for i in sorted(label_map.keys())]
        print(f"  {'':>12s}", end='')
        for n in names: print(f"  {n[:6]:>6s}", end='')
        print()
        for i, row in enumerate(test_r['cm']):
            print(f"  {names[i]:>12s}", end='')
            for v in row: print(f"  {v:>6d}", end='')
            print()

        lat = measure_latency(model, X_test)
        print(f"\n  Latency: mean={lat['mean_ms']:.4f}ms, p99={lat['p99_ms']:.4f}ms")

        model_path = os.path.join(models_dir, f'{model_key}_model.joblib')
        joblib.dump(model, model_path, compress=3)
        size_kb = os.path.getsize(model_path) / 1024
        size_mb = size_kb / 1024
        print(f"  Model size: {size_kb:.2f} KB ({size_mb:.4f} MB)")

        size_ok = size_mb < 50
        lat_ok = lat['p99_ms'] < 100
        print(f"  Constraints: Size<50MB {'PASS' if size_ok else 'FAIL'} | Latency<100ms {'PASS' if lat_ok else 'FAIL'}")

        all_results[model_key] = {
            'model': model, 'train_time': train_time,
            'test_accuracy': test_r['accuracy'], 'test_f1': test_r['f1_score'],
            'latency': lat, 'size_kb': size_kb, 'size_mb': size_mb,
            'model_path': model_path, 'passes': size_ok and lat_ok
        }

    # Summary
    print("\n\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n  {'Model':<25s} {'Accuracy':>10s} {'F1':>8s} {'Lat(ms)':>10s} {'Size(KB)':>10s} {'OK':>4s}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*4}")

    best_key, best_f1 = None, 0
    for k, r in all_results.items():
        n = MODEL_CONFIGS[k]['name']
        p = "Y" if r['passes'] else "N"
        print(f"  {n:<25s} {r['test_accuracy']:>9.4f} {r['test_f1']:>7.4f} {r['latency']['mean_ms']:>9.4f} {r['size_kb']:>9.2f} {p:>4s}")
        if r['passes'] and r['test_f1'] > best_f1:
            best_f1 = r['test_f1']
            best_key = k

    if best_key:
        print(f"\n  >> Best: {MODEL_CONFIGS[best_key]['name']}")
        best_path = os.path.join(models_dir, 'best_model.joblib')
        joblib.dump(all_results[best_key]['model'], best_path, compress=3)
        meta = {
            'model_type': best_key, 'name': MODEL_CONFIGS[best_key]['name'],
            'accuracy': all_results[best_key]['test_accuracy'],
            'f1': all_results[best_key]['test_f1'],
            'latency_ms': all_results[best_key]['latency']['mean_ms'],
            'size_kb': all_results[best_key]['size_kb'],
            'features': data['selected_features'],
            'label_map': label_map
        }
        joblib.dump(meta, os.path.join(models_dir, 'model_metadata.joblib'))
        print(f"    Saved to: {best_path}")

    print("\n  TRAINING COMPLETE [OK]")
    return all_results, data


if __name__ == "__main__":
    results, data = train_all_models()
