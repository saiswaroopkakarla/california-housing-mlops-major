import joblib
from sklearn.datasets import fetch_california_housing

def main():
    # Load model
    model = joblib.load("artifacts/linear_model.joblib")

    # Load test data
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Predict
    preds = model.predict(X[:5])
    print(" Sample Predictions:")
    for i, pred in enumerate(preds):
        print(f"  {i+1}: Predicted = {pred:.2f}, Actual = {y[i]:.2f}")

if __name__ == "__main__":
    main()

