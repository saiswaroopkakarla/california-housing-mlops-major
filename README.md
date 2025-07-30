
#  California Housing – MLOps Major Assignment

This project demonstrates a complete **MLOps pipeline** built using **Linear Regression** on the **California Housing dataset**. It covers model training, testing, manual quantization, Dockerization, and CI/CD automation using **GitHub Actions**.

---

##  Project Structure

```
.
├── src/                # Python source code for training, inference, quantization
│   ├── train.py
│   ├── quantize.py
│   └── predict.py
├── tests/              # Pytest unit tests
│   └── test_train.py
├── artifacts/          # Saved models and parameters (generated after training)
├── Dockerfile          # Docker config to build & run model container
├── .github/workflows/  # GitHub Actions CI/CD pipeline
│   └── ci.yml
├── requirements.txt    # Python dependencies
└── README.md
```



---

##  Objective

- Use `LinearRegression` from `scikit-learn` (only this model is allowed)
- Evaluate using R² Score and Mean Squared Error
- Save and quantize the model manually using `uint8` format
- Perform inference using dequantized parameters
- Package the model using Docker
- Automate testing, training, and containerization using GitHub Actions

---

## Model Comparison Table

| Model Stage               | Format   | File Name                 | Size  | R² Score | MSE     |
|---------------------------|----------|---------------------------|-------|----------|---------|
| Original Model            | float32  | `linear_model.joblib`     | 697 B | 0.6062   | 0.5243  |
| Quantized (Manual - uint8)| uint8    | `quant_params.joblib`     | 525 B | ~0.57    | 8.5391  |

---

##  How to Use

### 1. Train the Model

```bash
python src/train.py

```

### 2. Quantize the Model

```bash
python src/quantize.py

```

### 3. Run Predictions (inside Docker)

```bash
docker build -t housing-mlops .
docker run --rm housing-mlops
```
---
## Testing
Unit tests are written using pytest in the tests/ folder.

To run tests:
```bash

pytest tests/test_train.py -vvs

```

### Tests include:

Dataset loading

Model instantiation

Training verification (coef_)

R² threshold validation

---

## GitHub Actions – CI/CD Pipeline
The .github/workflows/ci.yml automates the full MLOps pipeline with three jobs:
```
Job Name                             Description
test-suite                   Runs pytest to validate training code
train-and-quantize           Trains model and performs quantization
build-and-test-container     Builds Docker image and runs predictions inside it
```

Runs automatically on every push to the main branch.

##  Requirements
Install dependencies using:

```bash
pip install -r requirements.txt
```
### Required Python Libraries:
```bash
scikit-learn

joblib

pytest
```
