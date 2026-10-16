"""Train and save the house-price regression pipeline."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "indian.csv"
MODEL_PATH = BASE_DIR / "house_price_model.joblib"
TARGET = "MEDV_LAKH"


def main():
    data = pd.read_csv(DATA_PATH)
    X = data.drop(columns=[TARGET])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=300, random_state=42, min_samples_leaf=2, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, predictions):.3f} INR lakh")
    print(f"R²:  {r2_score(y_test, predictions):.3f}")
    joblib.dump({"model": model, "features": list(X.columns), "target": TARGET}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
