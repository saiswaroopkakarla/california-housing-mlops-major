import pytest
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib
import os

# Test 1: Dataset loads properly
def test_dataset_load():
    data = fetch_california_housing()
    assert data.data.shape[0] > 0
    assert data.target.shape[0] > 0

# Test 2: Model instance is correct
def test_model_instance():
    model = LinearRegression()
    assert isinstance(model, LinearRegression)

# Test 3: Model is trained (coef_ should exist)
def test_model_training():
    data = fetch_california_housing()
    X, y = data.data, data.target
    model = LinearRegression()
    model.fit(X, y)
    assert hasattr(model, "coef_")

# Test 4: R2 score is above threshold
def test_model_r2_threshold():
    data = fetch_california_housing()
    X, y = data.data, data.target
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    assert r2 > 0.5

