
import pandas as pd
import streamlit as st
import plotly.express as px

from database import (
    init_db,
    save_prediction,
    get_prediction_history,
)
from train_model import (
    ensure_dataset,
    train_or_load_model,
    FEATURES,
)

st.set_page_config(
    page_title="Anime AI | Wine Quality Explorer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# LIGHT ANIME THEME: HEADER AND BODY ARE SEPARATE
# ==================================================
st.markdown("""
<style>
/* MAIN BODY */
.stApp {
    background: linear-gradient(
        135deg, #f5f3ff 0%, #eef7ff 52%, #fdf2f8 100%
    );
    color: #242044;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* SEPARATE LIGHT PURPLE HEADER */
.hero-banner {
    background: linear-gradient(
        115deg, #e9d5ff 0%, #ddd6fe 48%, #fbcfe8 100%
    );
    border: 1px solid #c084fc;
    border-radius: 24px;
    padding: 30px;
    margin-bottom: 26px;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.12);
}

.hero-tag {
    color: #6d28d9;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 3px;
    margin-bottom: 12px;
}

.hero-banner h1 {
    color: #35105f !important;
    font-size: 2.35rem;
    margin: 0;
}

.hero-banner p {
    color: #44315f !important;
    font-size: 1.05rem;
    line-height: 1.7;
    margin: 12px 0 0 0;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ede9fe, #f5f3ff);
    border-right: 1px solid #d8b4fe;
}

/* GENERAL TEXT */
h1, h2, h3 {
    color: #5b21b6 !important;
}

p, label, .stMarkdown {
    color: #30264a;
}

/* METRIC CARDS */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #ddd6fe;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 4px 15px rgba(91, 33, 182, 0.07);
    transition: 0.2s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: #a78bfa;
    box-shadow: 0 5px 18px rgba(124, 58, 237, 0.13);
}

div[data-testid="stMetricLabel"] {
    color: #6d28d9 !important;
}

div[data-testid="stMetricValue"] {
    color: #251342 !important;
}

/* TABS */
div[data-testid="stTabs"] button[role="tab"] {
    color: #5b3b85;
    border-radius: 10px 10px 0 0;
}

div[data-testid="stTabs"] button[role="tab"]:hover {
    color: #7c3aed;
    background: #ede9fe;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #6d28d9 !important;
    border-bottom: 3px solid #a855f7 !important;
}

/* PREDICTION FORM CARD */
div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid #d8b4fe;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 5px 20px rgba(91, 33, 182, 0.06);
}

/* INPUT VALUES: DARK TEXT */
div[data-baseweb="input"] input,
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {
    color: #21143d !important;
    -webkit-text-fill-color: #21143d !important;
    background-color: #ffffff !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

div[data-baseweb="input"] {
    background-color: #ffffff !important;
    border: 1px solid #c4b5fd !important;
    border-radius: 10px !important;
}

.stNumberInput label,
.stSelectbox label,
.stRadio label,
.stMultiSelect label {
    color: #45315f !important;
}

/* INPUT STEPPER BUTTONS */
.stNumberInput button {
    color: #6d28d9 !important;
}

/* BUTTONS */
.stButton > button,
.stDownloadButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(100deg, #8b5cf6, #a855f7, #ec4899);
    color: #ffffff !important;
    border: 1px solid #a78bfa;
    border-radius: 12px;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    box-shadow: 0 4px 16px rgba(168, 85, 247, 0.25);
    border-color: #7c3aed;
}

/* INFO BOX: PALE BLUE WITH DARK TEXT */
div[data-testid="stAlert"] {
    background: #e0f2fe !important;
    border: 1px solid #7dd3fc !important;
    border-radius: 12px;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
    color: #12304a !important;
}

/* TABLE */
div[data-testid="stDataFrame"] {
    border: 1px solid #ddd6fe;
    border-radius: 12px;
    overflow: hidden;
}

/* DIVIDERS */
hr {
    border-color: #ddd6fe;
}

/* FOOTER */
.anime-footer {
    text-align: center;
    color: #7c3aed;
    padding: 24px 0 6px;
    font-size: 0.85rem;
    letter-spacing: 1px;
}

/* MOBILE */
@media (max-width: 768px) {
    .hero-banner {
        padding: 20px;
    }

    .hero-banner h1 {
        font-size: 1.7rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-tag">✦ NEXT-GEN MACHINE LEARNING LAB ✦</div>
    <h1>🧪 Wine Quality Explorer</h1>
    <p>
        Decode wine chemistry, explore data patterns, and predict
        estimated quality using a Random Forest machine learning model.
    </p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# DATA AND MODEL
# ==================================================
@st.cache_resource
def setup_project():
    init_db()
    data = ensure_dataset()
    model, metrics = train_or_load_model(data)
    return data, model, metrics

try:
    data, model, metrics = setup_project()
except Exception as exc:
    st.error("The app could not prepare its dataset or model.")
    st.write("Check your dataset files, model setup, and packages.")
    st.exception(exc)
    st.stop()

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("## 🌸 ANIME AI LAB")
    st.caption("WINE CHEMISTRY ANALYTICS")
    st.divider()

    st.markdown("### 🧬 About")
    st.write(
        "An educational machine learning project exploring "
        "wine chemistry and recorded quality scores."
    )

    st.divider()
    st.markdown("### 📡 Model Statistics")
    st.metric("Dataset records", f"{len(data):,}")
    st.metric("Test MAE", f"{metrics['mae']:.2f}")
    st.metric("Test R²", f"{metrics['r2']:.3f}")

    st.caption("MAE measures average prediction error.")
    st.caption("R² summarizes model fit on test data.")

# ==================================================
# TABS
# ==================================================
tab_predict, tab_explore, tab_history, tab_about = st.tabs([
    "🔬 Predict Quality",
    "📊 Explore Dataset",
    "🗂️ Prediction History",
    "ℹ️ Project Details",
])

# ==================================================
# PREDICTION
# ==================================================
with tab_predict:
    st.subheader("🔮 Wine Quality Prediction")
    st.write(
        "Enter the 11 chemical measurements to obtain "
        "a model-based estimated quality score."
    )

    wine_type = st.radio(
        "Choose wine color",
        ["Red", "White"],
        horizontal=True,
        key="wine_color",
    )

    st.info(
        "Enter measurements in the units used by the dataset. "
        "The model requires all 11 chemistry features."
    )

    defaults = {
        "fixed acidity": 7.0,
        "volatile acidity": 0.50,
        "citric acid": 0.25,
        "residual sugar": 2.5,
        "chlorides": 0.07,
        "free sulfur dioxide": 15.0,
        "total sulfur dioxide": 45.0,
        "density": 0.996,
        "pH": 3.30,
        "sulphates": 0.60,
        "alcohol": 10.5,
    }

    ranges = {
        "fixed acidity": (4.0, 16.0, 0.1),
        "volatile acidity": (0.1, 1.6, 0.01),
        "citric acid": (0.0, 1.7, 0.01),
        "residual sugar": (0.5, 20.0, 0.1),
        "chlorides": (0.01, 0.3, 0.001),
        "free sulfur dioxide": (1.0, 80.0, 1.0),
        "total sulfur dioxide": (6.0, 300.0, 1.0),
        "density": (0.990, 1.005, 0.0001),
        "pH": (2.7, 4.2, 0.01),
        "sulphates": (0.2, 2.0, 0.01),
        "alcohol": (8.0, 15.0, 0.1),
    }

    with st.form("prediction_form"):
        cols = st.columns(3)
        values = {}

        for idx, feature in enumerate(FEATURES):
            lo, hi, step = ranges[feature]
            with cols[idx % 3]:
                values[feature] = st.number_input(
                    feature.title(),
                    min_value=float(lo),
                    max_value=float(hi),
                    value=float(defaults[feature]),
                    step=float(step),
                    format=(
                        "%.4f" if feature == "density"
                        else "%.3f" if feature == "chlorides"
                        else "%.2f"
                    ),
                    help=f"Dataset feature: {feature}",
                    key=f"input_{feature}",
                )

        submitted = st.form_submit_button(
            "✨ Predict Wine Quality",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        row = pd.DataFrame([
            {feature: values[feature] for feature in FEATURES}
        ])

        prediction = float(model.predict(row)[0])
        prediction = max(0.0, min(10.0, prediction))
        rounded_score = int(round(prediction))

        if rounded_score <= 4:
            grade = "Low"
        elif rounded_score == 5:
            grade = "Average"
        elif rounded_score == 6:
            grade = "Good"
        else:
            grade = "Very good"

        save_prediction(
            wine_type, values, prediction, rounded_score, grade
        )

        st.divider()
        st.markdown("### ✨ Prediction Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated score", f"{prediction:.2f} / 10")
        c2.metric("Rounded score", str(rounded_score))
        c3.metric("Estimated grade", grade)

        st.success("Prediction saved to the database.")
        st.caption(
            "This is a model estimate, not a laboratory result "
            "or official certification."
        )

# ==================================================
# DATASET EXPLORER
# ==================================================
with tab_explore:
    st.subheader("📊 Explore Wine Chemistry")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", f"{len(data):,}")
    c2.metric(
        "Red wine records",
        f"{int((data['wine_type'] == 'Red').sum()):,}",
    )
    c3.metric(
        "White wine records",
        f"{int((data['wine_type'] == 'White').sum()):,}",
    )

    st.divider()

    color_filter = st.multiselect(
        "Filter wine color",
        ["Red", "White"],
        default=["Red", "White"],
    )

    filtered = data[data["wine_type"].isin(color_filter)]

    if filtered.empty:
        st.warning("Select at least one wine color.")
    else:
        left, right = st.columns(2)

        with left:
            fig = px.histogram(
                filtered,
                x="quality",
                color="wine_type",
                barmode="overlay",
                title="Quality Score Distribution",
                labels={
                    "quality": "Recorded quality score",
                    "wine_type": "Wine color",
                },
                color_discrete_map={
                    "Red": "#e11d48",
                    "White": "#0891b2",
                },
                template="plotly_white",
            )
            fig.update_layout(
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(255,255,255,0.65)",
                title_font_color="#6d28d9",
                font_color="#30264a",
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.box(
                filtered,
                x="wine_type",
                y="alcohol",
                color="wine_type",
                title="Alcohol Measurements by Wine Color",
                labels={
                    "wine_type": "Wine color",
                    "alcohol": "Alcohol (% by volume)",
                },
                color_discrete_map={
                    "Red": "#e11d48",
                    "White": "#0891b2",
                },
                template="plotly_white",
            )
            fig.update_layout(
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(255,255,255,0.65)",
                title_font_color="#6d28d9",
                font_color="#30264a",
            )
            st.plotly_chart(fig, use_container_width=True)

        feature_choice = st.selectbox(
            "Explore a chemical feature",
            FEATURES,
            index=FEATURES.index("alcohol"),
        )

        sample = filtered.sample(
            n=min(1500, len(filtered)),
            random_state=42,
        )

        fig = px.scatter(
            sample,
            x=feature_choice,
            y="quality",
            color="wine_type",
            opacity=0.7,
            title=f"{feature_choice.title()} vs Recorded Quality",
            labels={
                "quality": "Recorded quality score",
                "wine_type": "Wine color",
            },
            color_discrete_map={
                "Red": "#e11d48",
                "White": "#0891b2",
            },
            template="plotly_white",
        )
        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0.65)",
            title_font_color="#6d28d9",
            font_color="#30264a",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📁 View dataset records"):
            st.dataframe(
                filtered.head(100),
                use_container_width=True,
                hide_index=True,
            )

# ==================================================
# PREDICTION HISTORY
# ==================================================
with tab_history:
    st.subheader("🗂️ Prediction History")
    history = get_prediction_history()

    if history.empty:
        st.info("No predictions saved yet. Use the prediction tab first.")
    else:
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.scatter(
            history,
            x="created_at",
            y="predicted_score",
            color="wine_type",
            hover_data=["rounded_score", "grade"],
            title="Prediction History Over Time",
            labels={
                "created_at": "Time",
                "predicted_score": "Predicted score",
                "wine_type": "Wine color",
            },
            color_discrete_map={
                "Red": "#e11d48",
                "White": "#0891b2",
            },
            template="plotly_white",
        )
        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0.65)",
            title_font_color="#6d28d9",
            font_color="#30264a",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Download Prediction History (CSV)",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

# ==================================================
# PROJECT DETAILS
# ==================================================
with tab_about:
    st.subheader("ℹ️ Inside the AI Laboratory")

    st.markdown("""
    ### 🧠 How the application works

    **01 — Dataset preparation**

    Loads and combines red and white wine chemistry records.

    **02 — Feature processing**

    Uses 11 numeric chemistry measurements as model inputs.

    **03 — Machine learning**

    A Random Forest regression model estimates quality scores.

    **04 — Model evaluation**

    MAE and R² summarize performance on held-out test data.

    **05 — Prediction history**

    Predictions are saved to SQLite for later review.

    ---

    ### 🧪 Scientific note

    The model learns patterns from historical human-assigned
    quality scores. Predictions are estimates, not laboratory
    measurements or official certification.
    """)

    st.markdown("### 🔬 Model input features")
    st.write(", ".join(FEATURES))

    st.markdown("### 📚 Dataset source")
    st.link_button(
        "Open UCI Wine Quality Dataset",
        "https://archive.ics.uci.edu/dataset/186/wine+quality",
    )

    st.caption(
        "Built with Python, Streamlit, Pandas, Plotly, "
        "scikit-learn, and SQLite."
    )

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<div class="anime-footer">
    ✦ WINE QUALITY AI LAB · EXPLORE DATA · DISCOVER PATTERNS ✦
</div>
""", unsafe_allow_html=True)
