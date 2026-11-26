
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="Wine Quality Explorer",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Wine Quality Chemistry Explorer & Grader")
st.write(
    "Explore physicochemical measurements, analyse wine quality "
    "and compare machine learning models."
)

st.info(
    "Predictions are estimates from historical dataset patterns. "
    "This application does not test, taste, or certify a physical sample."
)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    base = (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/wine-quality/"
    )

    red = pd.read_csv(base + "winequality-red.csv", sep=";")
    white = pd.read_csv(base + "winequality-white.csv", sep=";")

    red["wine_type"] = "Red"
    white["wine_type"] = "White"

    return pd.concat([red, white], ignore_index=True)

try:
    data = load_data()
except Exception as error:
    st.error("Could not load the UCI dataset. Check your internet connection and reload the app.")
    st.stop()

# ---------------- DATA OVERVIEW ----------------

st.header("1. Dataset Explorer")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total samples", f"{len(data):,}")
c2.metric("Chemical attributes", "11")
c3.metric("Red wine samples", f"{(data['wine_type'] == 'Red').sum():,}")
c4.metric("White wine samples", f"{(data['wine_type'] == 'White').sum():,}")

wine_type = st.selectbox(
    "Choose dataset",
    ["Both", "Red", "White"]
)

filtered = data.copy()
if wine_type != "Both":
    filtered = filtered[filtered["wine_type"] == wine_type]

st.subheader("Dataset preview")
st.dataframe(filtered.head(20), use_container_width=True)

st.subheader("Summary statistics")
st.dataframe(
    filtered.drop(columns="wine_type").describe().round(3),
    use_container_width=True
)

# ---------------- CHEMISTRY EXPLORER ----------------

st.header("2. Chemistry Explorer")

numeric_columns = [
    column for column in data.columns
    if column not in ["quality", "wine_type"]
]

left, right = st.columns(2)

with left:
    feature = st.selectbox(
        "Select a chemical property",
        numeric_columns,
        index=numeric_columns.index("alcohol")
    )

    fig = px.histogram(
        filtered,
        x=feature,
        color="wine_type",
        marginal="box",
        title=f"Distribution of {feature}"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    fig = px.box(
        filtered,
        x="quality",
        y=feature,
        color="wine_type",
        title=f"{feature} across quality scores"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Correlation heatmap")

corr = filtered.drop(columns="wine_type").corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Relationships between measured properties"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- MODEL TRAINING ----------------

st.header("3. Machine Learning")

st.write(
    "The models learn from chemical measurements and the quality scores "
    "provided in the UCI dataset. We hold out test samples for evaluation."
)

@st.cache_resource
def train_models():
    X = data[numeric_columns]
    y = data["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predictions),
            "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
            "R²": r2_score(y_test, predictions)
        })

    return models, pd.DataFrame(results)

with st.spinner("Training models and evaluating test data..."):
    models, results = train_models()

st.subheader("Model performance")

st.dataframe(
    results.round(3),
    hide_index=True,
    use_container_width=True
)

st.caption(
    "MAE and RMSE measure prediction error; lower is generally better. "
    "R² measures how well predictions explain variation in test scores; "
    "higher is generally better."
)

fig = px.bar(
    results.melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE"],
        var_name="Metric",
        value_name="Error"
    ),
    x="Model",
    y="Error",
    color="Metric",
    barmode="group",
    title="Compare model errors"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- FEATURE IMPORTANCE ----------------

st.subheader("Feature importance")

importance_model = models["Random Forest"]

importance = pd.DataFrame({
    "Chemical property": numeric_columns,
    "Importance": importance_model.feature_importances_
}).sort_values("Importance", ascending=False)

fig = px.bar(
    importance.sort_values("Importance"),
    x="Importance",
    y="Chemical property",
    orientation="h",
    title="Random Forest feature importance"
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Feature importance describes how the model uses features. "
    "It does not prove that a chemical property causes higher quality."
)

# ---------------- QUALITY GRADER ----------------

st.header("4. Wine Quality Grader")

st.write(
    "Enter laboratory-measured chemical values. "
    "The initial values below are example inputs, not a real sample."
)

defaults = data[numeric_columns].median()

with st.form("prediction_form"):
    st.subheader("Chemical measurements")

    input_values = {}

    columns = st.columns(3)

    for i, feature_name in enumerate(numeric_columns):
        with columns[i % 3]:
            input_values[feature_name] = st.number_input(
                feature_name.replace("_", " ").title(),
                min_value=0.0,
                value=float(defaults[feature_name]),
                step=0.1,
                format="%.4f"
            )

    selected_model = st.selectbox(
        "Choose prediction model",
        ["Random Forest", "Gradient Boosting"]
    )

    submitted = st.form_submit_button("Predict Wine Quality")

if submitted:
    input_df = pd.DataFrame(
        [input_values],
        columns=numeric_columns
    )

    score = float(models[selected_model].predict(input_df)[0])

    # Round the estimate to the nearest integer score.
    rounded_score = int(np.clip(np.rint(score), 0, 10))

    st.subheader("Prediction result")

    result_col1, result_col2 = st.columns(2)

    result_col1.metric(
        "Estimated quality score",
        f"{score:.2f} / 10"
    )

    result_col2.metric(
        "Nearest integer score",
        rounded_score
    )

    if rounded_score <= 4:
        grade = "Lower predicted score"
    elif rounded_score <= 6:
        grade = "Middle predicted score"
    else:
        grade = "Higher predicted score"

    st.write(f"**Indicative grade:** {grade}")
    st.write(f"**Model used:** {selected_model}")

    st.warning(
        "This is a model estimate, not a laboratory result, "
        "a sensory evaluation, or a guarantee of safety. "
        "Inputs outside the dataset's range may produce unreliable results."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Dataset: UCI Wine Quality. "
    "Model scores are predictions of dataset quality ratings, "
    "not chemical purity or food-safety certification."
)
