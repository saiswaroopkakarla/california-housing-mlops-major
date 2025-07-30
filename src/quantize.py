import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

"""def quantize(array):
    # Scale to 0–255
    min_val = array.min()
    max_val = array.max()
    scale = 255 / (max_val - min_val)
    quantized = ((array - min_val) * scale).astype(np.uint8)
    return quantized, scale, min_val"""

def quantize(array):
    min_val = np.min(array)
    max_val = np.max(array)

    if max_val - min_val == 0:
        quantized = np.zeros_like(array, dtype=np.uint8)
        scale = 1
    else:
        scale = 255 / (max_val - min_val)
        quantized = np.round((array - min_val) * scale).astype(np.uint8)

    return quantized, min_val, max_val


def dequantize(quantized, min_val, max_val):
    if max_val - min_val == 0:
        return np.full_like(quantized, fill_value=min_val, dtype=np.float32)
    else:
        scale = (max_val - min_val) / 255
        return (quantized.astype(np.float32) * scale) + min_val

def main():
    # Load trained model
    model = joblib.load("artifacts/linear_model.joblib")
    coef = model.coef_
    intercept = model.intercept_

    # Save raw params
    raw_params = {"coef": coef, "intercept": intercept}
    joblib.dump(raw_params, "artifacts/unquant_params.joblib")
    print("Saved unquantized parameters")

    # Quantize
    q_coef, coef_scale, coef_min = quantize(coef)
    q_intercept, int_scale, int_min = quantize(np.array([intercept]))

    quant_params = {
        "q_coef": q_coef,
        "q_intercept": q_intercept,
        "scales": {"coef": coef_scale, "intercept": int_scale},
        "mins": {"coef": coef_min, "intercept": int_min}
    }

    joblib.dump(quant_params, "artifacts/quant_params.joblib")
    print("Saved quantized parameters")

    # Inference using dequantized weights
    data = fetch_california_housing()
    X, y = data.data, data.target

    d_coef = dequantize(q_coef, coef_scale, coef_min)
    d_intercept = dequantize(q_intercept, int_scale, int_min)[0]
    y_pred = X @ d_coef + d_intercept

    loss = mean_squared_error(y, y_pred)
    print(f"Inference MSE with quantized weights: {loss:.4f}")

if __name__ == "__main__":
    main()

