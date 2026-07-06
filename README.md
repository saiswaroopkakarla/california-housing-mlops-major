# California Housing — End-to-End MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LinearRegression-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-containerised-2496ED?logo=docker&logoColor=white)
![CI/CD](https://img.shields.io/badge/GitHub%20Actions-3--job%20pipeline-2088FF?logo=githubactions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A complete MLOps lifecycle implementation on the California Housing dataset — from model training through manual uint8 quantization, pytest unit testing, Docker containerisation, and a 3-job GitHub Actions CI/CD pipeline.

---

## Pipeline Overview

```
[1] Test Suite          [2] Train & Quantize         [3] Build & Run Container
────────────────        ────────────────────         ──────────────────────────
pytest tests/     →→→  python src/train.py     →→→  docker build
4 unit tests           python src/quantize.py        docker run
R² threshold           artifacts/ uploaded            src/predict.py executes
assertion              as CI artifact                 inside container
```

All 3 jobs run sequentially on every push to `main` — job 2 only runs if tests pass, job 3 only runs if training succeeds.

---

## Model Results

| Stage | Format | File | Size | R² Score | MSE |
|---|---|---|---|---|---|
| Original model | float32 | `artifacts/linear_model.joblib` | 697 B | 0.6062 | 0.5243 |
| Quantized (manual uint8) | uint8 | `artifacts/quant_params.joblib` | 525 B | ~0.57 | 8.5391 |

**Size reduction:** 697 B → 525 B (~25% smaller)  
**R² trade-off:** 0.6062 → ~0.57 (acceptable precision loss for the compression gain)

The quantization maps float32 weights to the 0–255 uint8 range using:
```
quantized = round((value - min) × 255 / (max - min))
dequantized = quantized × (max - min) / 255 + min
```

---

## Repository Structure

```
california-housing-mlops-major/
├── src/
│   ├── train.py        # Train LinearRegression, save to artifacts/
│   ├── quantize.py     # Manual uint8 quantization + dequantization
│   └── predict.py      # Load model, run inference on 5 samples
├── tests/
│   └── test_train.py   # 4 pytest tests incl. R² > 0.5 threshold assertion
├── artifacts/
│   ├── linear_model.joblib      # Trained float32 model
│   ├── quant_params.joblib      # Quantized uint8 weights
│   └── unquant_params.joblib    # Raw parameters (pre-quantization)
├── .github/workflows/
│   └── ci.yml          # 3-job CI/CD pipeline
├── Dockerfile          # python:3.10-slim, runs predict.py
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/saiswaroopkakarla/california-housing-mlops-major.git
cd california-housing-mlops-major
pip install -r requirements.txt
```

---

## Run Locally

```bash
# 1. Train — saves artifacts/linear_model.joblib
python src/train.py

# 2. Quantize — saves artifacts/quant_params.joblib
python src/quantize.py

# 3. Predict (5 sample predictions)
python src/predict.py

# 4. Run tests
pytest tests/test_train.py -v
```

**Expected test output:**
```
test_dataset_load        PASSED
test_model_instance      PASSED
test_model_training      PASSED
test_model_r2_threshold  PASSED
```

---

## Docker

```bash
# Build image
docker build -t housing-mlops .

# Run container (executes src/predict.py)
docker run --rm housing-mlops
```

**Expected container output:**
```
Sample Predictions:
  1: Predicted = 4.15, Actual = 4.53
  2: Predicted = 3.75, Actual = 3.58
  ...
```

---

## CI/CD Pipeline (GitHub Actions)

Defined in `.github/workflows/ci.yml`, triggers on every push to `main`:

| Job | Depends on | Steps |
|---|---|---|
| `test-suite` | — | Checkout → Python 3.10 → Install deps → `pytest tests/` |
| `train-and-quantize` | `test-suite` ✅ | Checkout → Install deps → `train.py` → `quantize.py` → Upload artifacts |
| `build-and-test-container` | `train-and-quantize` ✅ | Checkout → Docker Buildx → `docker build` → `docker run` |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| scikit-learn | Linear Regression model |
| NumPy | Manual uint8 quantization math |
| joblib | Model serialisation |
| pytest | Unit testing (4 tests) |
| Docker | Containerised inference |
| GitHub Actions | CI/CD automation |

---

## Author

**Kakarla Sai Swaroop** — M25DE1023, IIT Jodhpur M.Tech Data Engineering
