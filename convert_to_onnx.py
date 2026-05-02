import sys, traceback, io

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

log = open('convert_log.txt', 'w', encoding='utf-8')

def p(msg):
    print(msg)
    log.write(msg + '\n')
    log.flush()

try:
    import joblib
    import numpy as np
    import json
    import os

    model = joblib.load('models/best_model.joblib')
    artifacts = joblib.load('models/preprocessing_artifacts.joblib')
    meta = joblib.load('models/model_metadata.joblib')

    scaler = artifacts['scaler']
    selector = artifacts['selector']
    label_map = artifacts['label_map']
    all_features = artifacts['all_features']
    selected_features = artifacts['selected_features']
    n_features = len(selected_features)

    p("Model: " + type(model).__name__)
    p("Features: " + str(n_features))

    np.save('models/scaler_mean.npy', scaler.mean_)
    np.save('models/scaler_scale.npy', scaler.scale_)
    p("Saved scaler params")

    np.save('models/selector_support.npy', selector.get_support())
    p("Saved selector support")

    config = {
        'label_map': {str(k): v for k, v in label_map.items()},
        'all_features': all_features,
        'selected_features': selected_features,
        'model_name': meta.get('name', 'Gradient Boosting'),
    }
    with open('models/model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    p("Saved config")

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        p("skl2onnx imported OK")
        initial_type = [('input', FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
        with open('models/best_model.onnx', 'wb') as f:
            f.write(onnx_model.SerializeToString())
        p("ONNX saved: " + str(os.path.getsize('models/best_model.onnx') // 1024) + " KB")
    except Exception as e:
        p("skl2onnx conversion error: " + str(e))
        tb = traceback.format_exc()
        p(tb)

except Exception as e:
    tb = traceback.format_exc()
    p("FATAL: " + tb)

log.close()
