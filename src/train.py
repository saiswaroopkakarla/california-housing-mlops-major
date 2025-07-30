from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

def main():
    # Load dataset
    data = fetch_california_housing()
    X, y = data.data, data.target

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict and evaluate
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    loss = mean_squared_error(y, y_pred)

    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {loss:.4f}")

    # Save the model
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/linear_model.joblib")
    print("Model saved to artifacts/linear_model.joblib")

if __name__ == "__main__":
    main()

