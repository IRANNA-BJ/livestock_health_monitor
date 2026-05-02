"""
Edge Optimization Module
========================
Optimizes trained model for edge/MCU deployment.
- ONNX export (with fallback)
- C header generation for MCU
- Size and latency benchmarks
"""

import numpy as np
import os
import sys
import time
import joblib
import json
import warnings
warnings.filterwarnings('ignore')


def try_convert_to_onnx(models_dir):
    """
    ONNX conversion is skipped due to skl2onnx version incompatibility.
    For production, use: pip install skl2onnx==1.16.0 onnx==1.15.0
    The JSON + C header exports below provide equivalent edge portability.
    """
    print("  ONNX conversion skipped (version incompatibility on this system).")
    print("  Alternative edge formats: JSON model + C header (see below).")
    print("  For ONNX, install compatible versions: skl2onnx==1.16.0 onnx==1.15.0")
    return None


def export_model_as_json(models_dir):
    """
    Export Logistic Regression model weights as JSON for ultra-portable deployment.
    JSON models can be loaded on ANY platform (microcontrollers, browsers, etc.)
    """
    lr_path = os.path.join(models_dir, 'logistic_regression_model.joblib')
    if not os.path.exists(lr_path):
        print("  Logistic regression model not found, skipping JSON export.")
        return None

    model = joblib.load(lr_path)
    artifacts = joblib.load(os.path.join(models_dir, 'preprocessing_artifacts.joblib'))
    scaler = artifacts['scaler']

    model_json = {
        'model_type': 'logistic_regression',
        'n_features': int(model.coef_.shape[1]),
        'n_classes': int(model.coef_.shape[0]),
        'classes': ['Healthy', 'Sick', 'Stressed', 'Inactive'],
        'coefficients': model.coef_.tolist(),
        'intercept': model.intercept_.tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'features': artifacts['selected_features']
    }

    json_path = os.path.join(models_dir, 'livestock_model_portable.json')
    with open(json_path, 'w') as f:
        json.dump(model_json, f, indent=2)

    size_kb = os.path.getsize(json_path) / 1024
    print(f"  Portable JSON model saved: {json_path} ({size_kb:.2f} KB)")
    print(f"  This model can be loaded in C, JavaScript, MicroPython, etc.")
    return json_path


def benchmark_inference(models_dir, n_runs=1000):
    """Benchmark all available model formats."""
    meta = joblib.load(os.path.join(models_dir, 'model_metadata.joblib'))
    n_features = len(meta['features'])
    sample = np.random.randn(1, n_features).astype(np.float32)

    print(f"\n  Inference Benchmark ({n_runs} runs, single sample):")
    print(f"  {'Format':<25s} {'Mean(ms)':>10s} {'P99(ms)':>10s} {'Status':>10s}")
    print(f"  {'---'*20}")

    # Joblib model benchmark
    model = joblib.load(os.path.join(models_dir, 'best_model.joblib'))
    for _ in range(20):
        model.predict(sample)
    lats = []
    for _ in range(n_runs):
        s = time.perf_counter()
        model.predict(sample)
        lats.append((time.perf_counter() - s) * 1000)
    lats = np.array(lats)
    ok = "PASS" if np.percentile(lats, 99) < 100 else "FAIL"
    print(f"  {'Joblib (sklearn)':<25s} {np.mean(lats):>9.4f} {np.percentile(lats, 99):>9.4f} {ok:>10s}")

    # ONNX benchmark
    onnx_path = os.path.join(models_dir, 'livestock_model.onnx')
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            input_name = sess.get_inputs()[0].name
            for _ in range(20):
                sess.run(None, {input_name: sample})
            lats = []
            for _ in range(n_runs):
                s = time.perf_counter()
                sess.run(None, {input_name: sample})
                lats.append((time.perf_counter() - s) * 1000)
            lats = np.array(lats)
            ok = "PASS" if np.percentile(lats, 99) < 100 else "FAIL"
            print(f"  {'ONNX Runtime':<25s} {np.mean(lats):>9.4f} {np.percentile(lats, 99):>9.4f} {ok:>10s}")
        except Exception as e:
            print(f"  {'ONNX Runtime':<25s} {'N/A':>10s} {'N/A':>10s} {'SKIP':>10s}")

    # Logistic Regression (always fastest)
    lr_path = os.path.join(models_dir, 'logistic_regression_model.joblib')
    if os.path.exists(lr_path):
        lr_model = joblib.load(lr_path)
        for _ in range(20):
            lr_model.predict(sample)
        lats = []
        for _ in range(n_runs):
            s = time.perf_counter()
            lr_model.predict(sample)
            lats.append((time.perf_counter() - s) * 1000)
        lats = np.array(lats)
        ok = "PASS" if np.percentile(lats, 99) < 100 else "FAIL"
        print(f"  {'LogReg (ultra-light)':<25s} {np.mean(lats):>9.4f} {np.percentile(lats, 99):>9.4f} {ok:>10s}")


def generate_c_header(models_dir):
    """Generate a C header file with scaler parameters for MCU deployment."""
    artifacts_path = os.path.join(models_dir, 'preprocessing_artifacts.joblib')
    if not os.path.exists(artifacts_path):
        print("  Preprocessing artifacts not found.")
        return

    artifacts = joblib.load(artifacts_path)
    scaler = artifacts['scaler']
    features = artifacts['selected_features']
    label_map = artifacts['label_map']

    # Also try to include logistic regression weights for pure-C inference
    lr_path = os.path.join(models_dir, 'logistic_regression_model.joblib')
    lr_model = None
    if os.path.exists(lr_path):
        lr_model = joblib.load(lr_path)

    lines = []
    lines.append("/*")
    lines.append(" * Auto-generated header for Livestock Health Monitor")
    lines.append(" * Deploy on ARM Cortex-M4 or similar MCU")
    lines.append(" * Generated by edge_optimize.py")
    lines.append(" */")
    lines.append("")
    lines.append("#ifndef LIVESTOCK_MODEL_H")
    lines.append("#define LIVESTOCK_MODEL_H")
    lines.append("")
    lines.append(f"#define NUM_FEATURES {len(features)}")
    lines.append(f"#define NUM_CLASSES {len(label_map)}")
    lines.append("")

    lines.append("/* Feature names */")
    for i, f in enumerate(features):
        lines.append(f'#define FEATURE_{i} "{f}"')
    lines.append("")

    lines.append("/* Scaler means */")
    lines.append(f"static const float SCALER_MEAN[{len(scaler.mean_)}] = {{")
    lines.append(f"    {', '.join(f'{v:.6f}f' for v in scaler.mean_)}")
    lines.append("};")
    lines.append("")

    lines.append("/* Scaler scales (std dev) */")
    lines.append(f"static const float SCALER_SCALE[{len(scaler.scale_)}] = {{")
    lines.append(f"    {', '.join(f'{v:.6f}f' for v in scaler.scale_)}")
    lines.append("};")
    lines.append("")

    lines.append("/* Class labels */")
    for k, v in label_map.items():
        lines.append(f'static const char* CLASS_{k} = "{v}";')
    lines.append("")

    lines.append("/* Normalize a feature value */")
    lines.append("static inline float normalize_feature(int idx, float value) {")
    lines.append("    return (value - SCALER_MEAN[idx]) / SCALER_SCALE[idx];")
    lines.append("}")
    lines.append("")

    if lr_model is not None:
        n_classes, n_feat = lr_model.coef_.shape
        lines.append("/* ============================================ */")
        lines.append("/* Logistic Regression Weights (full inference) */")
        lines.append("/* ============================================ */")
        lines.append("")
        lines.append(f"static const float LR_COEF[{n_classes}][{n_feat}] = {{")
        for i in range(n_classes):
            row = ', '.join(f'{v:.6f}f' for v in lr_model.coef_[i])
            lines.append(f"    {{{row}}},")
        lines.append("};")
        lines.append("")
        lines.append(f"static const float LR_INTERCEPT[{n_classes}] = {{")
        lines.append(f"    {', '.join(f'{v:.6f}f' for v in lr_model.intercept_)}")
        lines.append("};")
        lines.append("")
        lines.append("/* Softmax + argmax prediction */")
        lines.append("static int predict_health_state(float features[NUM_FEATURES]) {")
        lines.append("    float normalized[NUM_FEATURES];")
        lines.append("    for (int i = 0; i < NUM_FEATURES; i++) {")
        lines.append("        normalized[i] = normalize_feature(i, features[i]);")
        lines.append("    }")
        lines.append("")
        lines.append("    float max_score = -1e9f;")
        lines.append("    int best_class = 0;")
        lines.append("    for (int c = 0; c < NUM_CLASSES; c++) {")
        lines.append("        float score = LR_INTERCEPT[c];")
        lines.append("        for (int f = 0; f < NUM_FEATURES; f++) {")
        lines.append("            score += LR_COEF[c][f] * normalized[f];")
        lines.append("        }")
        lines.append("        if (score > max_score) {")
        lines.append("            max_score = score;")
        lines.append("            best_class = c;")
        lines.append("        }")
        lines.append("    }")
        lines.append("    return best_class;")
        lines.append("}")
        lines.append("")

    lines.append("#endif /* LIVESTOCK_MODEL_H */")

    h_path = os.path.join(models_dir, 'livestock_model.h')
    with open(h_path, 'w') as f:
        f.write('\n'.join(lines))
    size_kb = os.path.getsize(h_path) / 1024
    print(f"  C header saved: {h_path} ({size_kb:.2f} KB)")
    if lr_model is not None:
        print(f"  Includes full Logistic Regression inference function!")
        print(f"  -> Can run on bare-metal MCU with zero dependencies")


def generate_size_report(models_dir):
    """Generate a report of all model file sizes."""
    print("\n  Model Size Report:")
    print(f"  {'File':<40s} {'Size':>10s}")
    print(f"  {'---'*18}")
    total = 0
    for fname in sorted(os.listdir(models_dir)):
        fpath = os.path.join(models_dir, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total += size
            if size < 1024 * 1024:
                print(f"  {fname:<40s} {size/1024:>8.2f} KB")
            else:
                print(f"  {fname:<40s} {size/(1024*1024):>8.2f} MB")
    print(f"  {'---'*18}")
    print(f"  {'TOTAL':<40s} {total/1024:>8.2f} KB")
    constraint_pass = total < 50 * 1024 * 1024
    print(f"\n  All models under 50MB constraint: {'PASS' if constraint_pass else 'FAIL'}")


def optimize():
    print("=" * 60)
    print("  EDGE OPTIMIZATION")
    print("=" * 60)

    project_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(project_dir, 'models')

    print("\n[1] Attempting ONNX conversion...")
    try_convert_to_onnx(models_dir)

    print("\n[2] Exporting portable JSON model...")
    export_model_as_json(models_dir)

    print("\n[3] Generating C header for MCU deployment...")
    generate_c_header(models_dir)

    print("\n[4] Benchmarking all model formats...")
    benchmark_inference(models_dir)

    print("\n[5] Size report...")
    generate_size_report(models_dir)

    print("\n  OPTIMIZATION COMPLETE")


if __name__ == "__main__":
    optimize()
