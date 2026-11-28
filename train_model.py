
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

MODEL_PATH = MODEL_DIR / "wine_quality_random_forest.joblib"
METRICS_PATH = MODEL_DIR / "model_metrics.csv"

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

TARGET = "quality"


def read_wine_csv(path):
    """Read either comma-separated or semicolon-separated CSV files."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Detect the separator from the header.
    with open(path, "r", encoding="utf-8-sig") as file:
        header = file.readline()

    separator = ";" if header.count(";") > header.count(",") else ","
    data = pd.read_csv(path, sep=separator, encoding="utf-8-sig")

    # Remove accidental spaces from column names.
    data.columns = data.columns.astype(str).str.strip()
    return data


def ensure_dataset():
    """Load, validate, and combine the local red and white datasets."""
    red_path = DATA_DIR / "redwine.csv"
    white_path = DATA_DIR / "whitewine.csv"

    red = read_wine_csv(red_path)
    white = read_wine_csv(white_path)

    required_columns = FEATURES + [TARGET]

    for name, frame in [("Red", red), ("White", white)]:
        missing = [col for col in required_columns if col not in frame.columns]
        if missing:
            raise ValueError(
                f"{name} dataset has missing columns: {missing}. "
                f"Columns found: {frame.columns.tolist()}"
            )

    red["wine_type"] = "Red"
    white["wine_type"] = "White"

    data = pd.concat([red, white], ignore_index=True)

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=required_columns).reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid dataset rows remain after cleaning.")

    return data


def train_or_load_model(data, force_train=False):
    """Load an existing model or train and save a new Random Forest model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if (
        MODEL_PATH.exists()
        and METRICS_PATH.exists()
        and not force_train
    ):
        model = joblib.load(MODEL_PATH)
        metric_table = pd.read_csv(METRICS_PATH)

        if not {"mae", "r2"}.issubset(metric_table.columns) or metric_table.empty:
            force_train = True
        else:
            metrics = {
                "mae": float(metric_table.loc[0, "mae"]),
                "r2": float(metric_table.loc[0, "r2"]),
            }
            return model, metrics

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    joblib.dump(model, MODEL_PATH)
    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

    return model, metrics


if __name__ == "__main__":
    dataset = ensure_dataset()
    model, metrics = train_or_load_model(dataset, force_train=True)

    print(f"Total dataset rows: {len(dataset)}")
    print(f"Red wine rows: {(dataset['wine_type'] == 'Red').sum()}")
    print(f"White wine rows: {(dataset['wine_type'] == 'White').sum()}")
    print(f"Test MAE: {metrics['mae']:.3f}")
    print(f"Test R²: {metrics['r2']:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
