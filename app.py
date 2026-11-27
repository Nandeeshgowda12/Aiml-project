
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

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Anime AI | Wine Quality Explorer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# ANIME / FUTURISTIC THEME
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background:
            radial-gradient(
                circle at 5% 5%,
                rgba(139, 92, 246, 0.20),
                transparent 30%
            ),
            radial-gradient(
                circle at 95% 75%,
                rgba(6, 182, 212, 0.16),
                transparent 32%
            ),
            linear-gradient(
                135deg,
                #080b18 0%,
                #11132a 48%,
                #101d35 100%
            );
        color: #f4f2ff;
    }

    /* Main page spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #17112e 0%,
            #10172b 55%,
            #09101f 100%
        );
        border-right: 1px solid rgba(139, 92, 246, 0.45);
    }

    /* Main title and headings */
    h1, h2, h3 {
        color: #e3d5ff !important;
        letter-spacing: 0.3px;
    }

    h1 {
        text-shadow: 0 0 20px rgba(139, 92, 246, 0.45);
    }

    /* Paragraphs and labels */
    p, label, .stMarkdown {
        color: #e4e8ff;
    }

    /* Anime-inspired hero banner */
    .hero-banner {
        background: linear-gradient(
            120deg,
            rgba(124, 58, 237, 0.25),
            rgba(8, 145, 178, 0.18),
            rgba(219, 39, 119, 0.12)
        );
        border: 1px solid rgba(167, 139, 250, 0.55);
        border-radius: 22px;
        padding: 28px 30px;
        margin: 12px 0 24px 0;
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.12);
    }

    .hero-banner h1 {
        margin: 0;
        font-size: 2.1rem;
        color: #ffffff !important;
    }

    .hero-banner p {
        color: #cbd5ff;
        margin-top: 10px;
        margin-bottom: 0;
        font-size: 1rem;
    }

    .hero-tag {
        color: #67e8f9;
        font-size: 0.78rem;
        letter-spacing: 3px;
        font-weight: 800;
        text-transform: uppercase;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(38, 33, 75, 0.92),
            rgba(16, 31, 55, 0.92)
        );
        border: 1px solid rgba(139, 92, 246, 0.55);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.18);
        transition: all 0.25s ease;
    }

    div[data-testid="stMetric"]:hover {
        border-color: #67e8f9;
        box-shadow: 0 0 18px rgba(103, 232, 249, 0.17);
        transform: translateY(-2px);
    }

    div[data-testid="stMetricLabel"] {
        color: #c4b5fd !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* Tabs */
    div[data-testid="stTabs"] button[role="tab"] {
        color: #c4b5fd;
        border-radius: 10px 10px 0 0;
        padding: 12px 16px;
        transition: all 0.2s ease;
    }

    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: #67e8f9;
        background: rgba(139, 92, 246, 0.12);
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #67e8f9 !important;
        border-bottom: 3px solid #a78bfa !important;
        background: rgba(139, 92, 246, 0.10);
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(
            100deg,
            #7c3aed,
            #4f46e5,
            #0891b2
        );
        color: #ffffff !important;
        border: 1px solid rgba(196, 181, 253, 0.45);
        border-radius: 12px;
        font-weight: 700;
        transition: all 0.25s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        border-color: #67e8f9;
        box-shadow: 0 0 20px rgba(103, 232, 249, 0.30);
        transform: translateY(-2px);
    }

    /* Form containers */
    div[data-testid="stForm"] {
        background: rgba(18, 22, 48, 0.68);
        border: 1px solid rgba(139, 92, 246, 0.40);
        border-radius: 18px;
        padding: 22px;
    }

    /* Input fields */
    div[data-baseweb="input"],
    div[data-baseweb="select"],
    div[data-baseweb="textarea"] {
        border-radius: 10px;
    }

    
input, textarea {
    color: #17132b !important;
    -webkit-text-fill-color: #17132b !important;
    background-color: #ffffff !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] div {
    color: #202044 !important;
}

.stNumberInput label,
.stSelectbox label,
.stRadio label,
.stMultiSelect label {
    color: #e9ddff !important;
}


    /* Info, success and warning messages */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid rgba(139, 92, 246, 0.30);
    }

    /* Data tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(139, 92, 246, 0.38);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Dividers */
    hr {
        border-color: rgba(139, 92, 246, 0.35);
    }

    /* Small footer */
    .anime-footer {
        text-align: center;
        color: #a5b4fc;
        font-size: 0.85rem;
        padding: 22px 0 5px 0;
        letter-spacing: 0.5px;
    }

    /* Keep the design responsive on small screens */
    @media (max-width: 768px) {
        .hero-banner {
            padding: 20px;
        }

        .hero-banner h1 {
            font-size: 1.55rem;
        }

        .block-container {
            padding-top: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-tag">✦ NEXT-GEN MACHINE LEARNING LAB ✦</div>
        <h1>🧪 Wine Quality Explorer</h1>
        <p>
            Decode wine chemistry, explore data patterns, and predict
            estimated quality using a Random Forest machine learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# LOAD DATASET AND MODEL
# --------------------------------------------------
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
        "Please check that your data files, model code, "
        "and required packages are configured correctly."
    )
    st.exception(exc)
    st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🌌 AI LAB")
    st.caption("ANIME-INSPIRED ANALYTICS")

    st.divider()

    st.markdown("### 🧬 About this project")
    st.write(
        "An educational machine learning project exploring "
        "the relationship between wine chemistry and recorded quality."
    )

    st.divider()

    st.markdown("### 📡 Model status")
    st.metric("Dataset records", f"{len(data):,}")
    st.metric("Test MAE", f"{metrics['mae']:.2f}")
    st.metric("Test R²", f"{metrics['r2']:.3f}")

    st.caption(
        "MAE measures average prediction error. "
        "R² describes model fit on the test data."
    )

    st.divider()
    st.caption("⚡ Powered by Python · Streamlit · Random Forest")

# --------------------------------------------------
# NAVIGATION TABS
# --------------------------------------------------
tab_predict, tab_explore, tab_history, tab_about = st.tabs(
    [
        "🔬 Predict Quality",
        "📊 Explore Dataset",
        "🗂️ Prediction History",
        "ℹ️ Project Details",
    ]
)

# --------------------------------------------------
# TAB 1: PREDICTION
# --------------------------------------------------
with tab_predict:
    st.subheader("🔮 Wine Quality Prediction")
    st.write(
        "Enter the 11 chemical measurements below to get "
        "a model-based estimated quality score."
    )

    wine_type = st.radio(
        "Choose wine color",
        ["Red", "White"],
        horizontal=True,
        key="wine_color",
    )

    st.info(
        "Use the measurement units represented in the dataset. "
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

        c1.metric(
            "Estimated quality score",
            f"{prediction:.2f} / 10",
        )
        c2.metric(
            "Rounded score",
            str(rounded_score),
        )
        c3.metric(
            "Estimated grade",
            grade,
        )

        st.success("Prediction saved to the database.")

        st.caption(
            "These are model estimates based on historical dataset "
            "patterns, not laboratory results or official certification."
        )

# --------------------------------------------------
# TAB 2: DATA EXPLORATION
# --------------------------------------------------
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
                legend_title_text="Wine color",
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
            "Select a chemical feature",
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

# --------------------------------------------------
# TAB 3: PREDICTION HISTORY
# --------------------------------------------------
with tab_history:
    st.subheader("🗂️ Prediction History")
    st.write("Review your previously saved model predictions.")

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

# --------------------------------------------------
# TAB 4: PROJECT DETAILS
# --------------------------------------------------
with tab_about:
    st.subheader("ℹ️ Inside the AI Laboratory")

    st.markdown(
        """
        ### 🧠 How the application works

        **01 — Dataset preparation**
        
        Loads and combines the red and white wine chemistry records.

        **02 — Feature processing**
        
        Uses 11 numeric physicochemical measurements as model inputs.

        **03 — Machine learning**
        
        A Random Forest regression model estimates quality scores
        from patterns learned in the training data.

        **04 — Model evaluation**
        
        MAE and R² help summarize model performance on held-out data.

        **05 — Prediction history**
        
        Predictions are saved to SQLite for later review.

        ---

        ### 🧪 Important scientific note

        The model learns from historical, human-assigned quality scores.
        Its predictions are estimates and are not a substitute for
        professional evaluation or laboratory analysis.
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
        "Project built with Python, Streamlit, Pandas, Plotly, "
        "scikit-learn, and SQLite."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="anime-footer">
        ✦ WINE QUALITY AI LAB · EXPLORE DATA · DISCOVER PATTERNS ✦
    </div>
    """,
    unsafe_allow_html=True,
)
