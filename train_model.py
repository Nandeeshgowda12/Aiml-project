
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DB_PATH = ROOT / "wine_quality.db"

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


def ensure_dataset():
    """Load the red and white datasets from the local data folder."""

    red = pd.read_csv(red_path, sep=";")
white = pd.read_csv(white_path, sep=";")

    if not red_path.exists():
        raise FileNotFoundError(
            f"Red dataset not found: {red_path}"
        )

    if not white_path.exists():
        raise FileNotFoundError(
            f"White dataset not found: {white_path}"
        )

    # These CSV files are comma-separated.
    red = pd.read_csv(red_path)
    white = pd.read_csv(white_path)

    # Clean column names.
    red.columns = red.columns.astype(str).str.strip()
    white.columns = white.columns.astype(str).str.strip()

    # Identify each dataset's wine type.
    red["wine_type"] = "Red"
    white["wine_type"] = "White"

    # Combine both datasets.
    data = pd.concat(
        [red, white],
        ignore_index=True
    )

    # Check that all required columns exist.
    required_columns = FEATURES + [TARGET]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Convert the model inputs and target to numeric values.
    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Remove rows with missing required values.
    data = data.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "No valid rows found in the datasets."
        )

    return data


def train_or_load_model(data, force_train=False):
    """Train a Random Forest model or load the saved model."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if (
        MODEL_PATH.exists()
        and METRICS_PATH.exists()
        and not force_train
    ):
        model = joblib.load(MODEL_PATH)

        metric_table = pd.read_csv(METRICS_PATH)

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
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "mae": float(
            mean_absolute_error(y_test, predictions)
        ),
        "r2": float(
            r2_score(y_test, predictions)
        ),
    }

    joblib.dump(model, MODEL_PATH)

    pd.DataFrame([metrics]).to_csv(
        METRICS_PATH,
        index=False
    )

    return model, metrics


if __name__ == "__main__":
    dataset = ensure_dataset()

    trained_model, evaluation = train_or_load_model(
        dataset,
        force_train=True
    )

    print(f"Total dataset rows: {len(dataset)}")
    print(
        f"Red wine rows: "
        f"{(dataset['wine_type'] == 'Red').sum()}"
    )
    print(
        f"White wine rows: "
        f"{(dataset['wine_type'] == 'White').sum()}"
    )
    print(f"Test MAE: {evaluation['mae']:.3f}")
    print(f"Test R²: {evaluation['r2']:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
