
import pandas as pd
import streamlit as st
import plotly.express as px

from database import (
    init_db,
    save_prediction,
    get_prediction_history,
    get_dataset_summary,
)
from train_model import (
    ensure_dataset,
    train_or_load_model,
    FEATURES,
    TARGET,
    DATA_DIR,
)

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Anime AI | Wine Quality Explorer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================================================
# SEPARATE HEADER AND BODY THEME
# ==================================================
st.markdown(
    """
    <style>
    /* ---------- MAIN BODY BACKGROUND ---------- */
    .stApp {
        background:
            radial-gradient(
                circle at 90% 15%,
                rgba(14, 165, 233, 0.13),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #070b18 0%,
                #0d1530 55%,
                #101c35 100%
            );
        color: #f8fafc;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* ---------- SEPARATE HEADER BANNER ---------- */
    .hero-banner {
        background:
            radial-gradient(
                circle at 90% 15%,
                rgba(236, 72, 153, 0.30),
                transparent 35%
            ),
            linear-gradient(
                120deg,
                #32105c 0%,
                #512080 45%,
                #182d59 100%
            );
        border: 1px solid #a855f7;
        border-radius: 24px;
        padding: 32px;
        margin-bottom: 28px;
        box-shadow:
            0 0 22px rgba(168, 85, 247, 0.22),
            inset 0 0 24px rgba(236, 72, 153, 0.07);
    }

    .hero-tag {
        color: #67e8f9;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 3px;
        margin-bottom: 12px;
    }

    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 2.4rem;
        margin: 0;
        text-shadow: 0 0 16px rgba(216, 180, 254, 0.4);
    }

    .hero-banner p {
        color: #f3e8ff !important;
        font-size: 1.05rem;
        margin-top: 12px;
        margin-bottom: 0;
        line-height: 1.7;
    }

    /* ---------- SIDEBAR ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #1b1035,
            #10172b,
            #080d1b
        );
        border-right: 1px solid #58318a;
    }

    /* ---------- BODY HEADINGS ---------- */
    h1, h2, h3 {
        color: #d8b4fe !important;
    }

    h2, h3 {
        text-shadow: 0 0 10px rgba(139, 92, 246, 0.16);
    }

    p, label, .stMarkdown {
        color: #e2e8f0;
    }

    /* ---------- METRIC CARDS ---------- */
    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            #191d3c,
            #10243b
        );
        border: 1px solid #6741a5;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.20);
        transition: all 0.25s ease;
    }

    div[data-testid="stMetric"]:hover {
        border-color: #67e8f9;
        box-shadow: 0 0 18px rgba(103, 232, 249, 0.18);
        transform: translateY(-2px);
    }

    div[data-testid="stMetricLabel"] {
        color: #c4b5fd !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* ---------- TABS ---------- */
    div[data-testid="stTabs"] button[role="tab"] {
        color: #ddd6fe;
        border-radius: 10px 10px 0 0;
        padding: 12px 14px;
    }

    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: #67e8f9;
        background: rgba(139, 92, 246, 0.12);
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #67e8f9 !important;
        border-bottom: 3px solid #c084fc !important;
        background: rgba(139, 92, 246, 0.10);
    }

    /* ---------- FORM CARD ---------- */
    div[data-testid="stForm"] {
        background: linear-gradient(
            145deg,
            rgba(21, 27, 57, 0.96),
            rgba(13, 27, 48, 0.96)
        );
        border: 1px solid #513780;
        border-radius: 20px;
        padding: 22px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
    }

    /* ---------- INPUT VALUES: DARK TEXT ON WHITE ---------- */
    div[data-baseweb="input"] input,
    .stNumberInput input,
    .stTextInput input,
    .stTextArea textarea {
        color: #17132b !important;
        -webkit-text-fill-color: #17132b !important;
        background-color: #ffffff !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    div[data-baseweb="input"] {
        border: 1px solid #a78bfa;
        border-radius: 10px;
    }

    /* Input labels */
    .stNumberInput label,
    .stSelectbox label,
    .stRadio label,
    .stMultiSelect label {
        color: #e9ddff !important;
    }

    /* ---------- BUTTONS ---------- */
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(
            100deg,
            #9333ea,
            #6366f1,
            #0891b2
        );
        color: #ffffff !important;
        border: 1px solid #a78bfa;
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.25s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        border-color: #67e8f9;
        box-shadow: 0 0 18px rgba(103, 232, 249, 0.30);
        transform: translateY(-2px);
    }

    /* ---------- ALERTS ---------- */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* ---------- TABLES ---------- */
    div[data-testid="stDataFrame"] {
        border: 1px solid #513780;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- DIVIDERS ---------- */
    hr {
        border-color: rgba(167, 139, 250, 0.35);
    }

    /* ---------- FOOTER ---------- */
    .anime-footer {
        text-align: center;
        color: #a5b4fc;
        padding: 24px 0 6px;
        font-size: 0.85rem;
        letter-spacing: 1px;
    }

    /* ---------- MOBILE VIEW ---------- */
    @media (max-width: 768px) {
        .hero-banner {
            padding: 22px;
        }

        .hero-banner h1 {
            font-size: 1.7rem;
        }

        .hero-banner p {
            font-size: 0.95rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# HEADER SECTION
# ==================================================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-tag">
            ✦ NEXT-GEN MACHINE LEARNING LAB ✦
        </div>
        <h1>🧪 Wine Quality Explorer</h1>
        <p>
            Decode wine chemistry, explore data patterns, and predict
            estimated quality using a Random Forest machine learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# LOAD DATA AND MODEL
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
    st.write(
        "Check your data files, model configuration, "
        "and required Python packages."
    )
    st.exception(exc)
    st.stop()

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("## 🌌 ANIME AI LAB")
    st.caption("WINE CHEMISTRY ANALYTICS")
    st.divider()

    st.markdown("### 🧬 About")
    st.write(
        "An educational project exploring the relationship "
        "between wine chemistry and recorded quality scores."
    )

    st.divider()
    st.markdown("### 📡 Model Statistics")

    st.metric("Dataset records", f"{len(data):,}")
    st.metric("Test MAE", f"{metrics['mae']:.2f}")
    st.metric("Test R²", f"{metrics['r2']:.3f}")

    st.caption(
        "MAE measures average prediction error. "
        "R² summarizes model fit on test data."
    )

    st.divider()
    st.caption("Python · Streamlit · Random Forest")

# ==================================================
# NAVIGATION
# ==================================================
tab_predict, tab_explore, tab_history, tab_about = st.tabs(
    [
        "🔬 Predict Quality",
        "📊 Explore Dataset",
        "🗂️ Prediction History",
        "ℹ️ Project Details",
    ]
)

# ==================================================
# TAB 1: PREDICT QUALITY
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
                        "%.4f"
                        if feature == "density"
                        else "%.3f"
                        if feature == "chlorides"
                        else "%.2f"
                    ),
                    help=f"Dataset feature: {feature}",
                    key=f"input_{feature}",
                )

        submitted = st.form_submit_button(
            "⚡ Predict Wine Quality",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        row = pd.DataFrame(
            [{feature: values[feature] for feature in FEATURES}]
        )

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
            wine_type,
            values,
            prediction,
            rounded_score,
            grade,
        )

        st.divider()
        st.markdown("### ✨ Prediction Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated score", f"{prediction:.2f} / 10")
        c2.metric("Rounded score", str(rounded_score))
        c3.metric("Estimated grade", grade)

        st.success("Prediction saved to the database.")
        st.caption(
            "The grade is a project-defined interpretation of the "
            "model's estimate, not official certification."
        )

# ==================================================
# TAB 2: EXPLORE DATASET
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
                    "Red": "#fb7185",
                    "White": "#67e8f9",
                },
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(12,16,35,0.65)",
                title_font_color="#d8b4fe",
                font_color="#e4e8ff",
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
                    "Red": "#fb7185",
                    "White": "#67e8f9",
                },
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(12,16,35,0.65)",
                title_font_color="#d8b4fe",
                font_color="#e4e8ff",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

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
            opacity=0.70,
            title=f"{feature_choice.title()} vs Recorded Quality",
            labels={
                "quality": "Recorded quality score",
                "wine_type": "Wine color",
            },
            color_discrete_map={
                "Red": "#fb7185",
                "White": "#67e8f9",
            },
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,16,35,0.65)",
            title_font_color="#d8b4fe",
            font_color="#e4e8ff",
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📁 View dataset records"):
            st.dataframe(
                filtered.head(100),
                use_container_width=True,
                hide_index=True,
            )

# ==================================================
# TAB 3: PREDICTION HISTORY
# ==================================================
with tab_history:
    st.subheader("🗂️ Prediction History")
    st.write("Review previously saved model predictions.")

    history = get_prediction_history()

    if history.empty:
        st.info(
            "No predictions saved yet. "
            "Make a prediction in the first tab."
        )
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
                "Red": "#fb7185",
                "White": "#67e8f9",
            },
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(12,16,35,0.65)",
            title_font_color="#d8b4fe",
            font_color="#e4e8ff",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Download Prediction History (CSV)",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

# ==================================================
# TAB 4: PROJECT DETAILS
# ==================================================
with tab_about:
    st.subheader("ℹ️ Inside the AI Laboratory")

    st.markdown(
        """
        ### 🧠 How the application works

        **01 — Dataset preparation**

        Loads and combines red and white wine chemistry records.

        **02 — Feature processing**

        Uses 11 numeric physicochemical measurements as model inputs.

        **03 — Machine learning**

        A Random Forest regression model estimates quality scores
        from patterns learned in historical records.

        **04 — Model evaluation**

        MAE and R² summarize performance on held-out test data.

        **05 — Prediction history**

        Predictions are saved to SQLite for later review.

        ---

        ### 🧪 Scientific note

        Quality scores in the source dataset are human-assigned.
        Predictions are estimates, not laboratory measurements or
        official wine certifications.
        """
    )

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
st.markdown(
    """
    <div class="anime-footer">
        ✦ WINE QUALITY AI LAB · EXPLORE DATA · DISCOVER PATTERNS ✦
    </div>
    """,
    unsafe_allow_html=True,
)
