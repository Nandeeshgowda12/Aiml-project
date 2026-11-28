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
    page_title="Wine Quality Explorer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# RESPONSIVE DARK THEME
# --------------------------------------------------
st.markdown(
    """
    <style>
    /* ==========================================
       GLOBAL THEME
       ========================================== */

    :root {
        color-scheme: dark;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at top left,
                rgba(124, 58, 237, 0.20),
                transparent 42%
            ),
            radial-gradient(
                ellipse at bottom right,
                rgba(8, 145, 178, 0.15),
                transparent 40%
            ),
            linear-gradient(
                135deg,
                #0b1020 0%,
                #11152b 55%,
                #10243a 100%
            );

        color: #f1f5ff;
    }

    /* Keep content comfortable on laptops and phones */
    .block-container {
        width: 100%;
        max-width: 1500px;
        padding: 1.5rem 2rem 3rem 2rem;
    }

    /* ==========================================
       HEADINGS AND TEXT
       ========================================== */

    h1, h2, h3, h4 {
        color: #e9ddff !important;
    }

    p, li {
        color: #e2e8f0;
    }

    label,
    [data-testid="stWidgetLabel"] p {
        color: #f1eaff !important;
    }

    .stCaption,
    [data-testid="stCaptionContainer"] p {
        color: #bdc9e3 !important;
    }

    /* ==========================================
       HERO BANNER
       ========================================== */

    .hero-banner {
        background: linear-gradient(
            120deg,
            rgba(109, 40, 217, 0.30),
            rgba(8, 145, 178, 0.20),
            rgba(190, 24, 93, 0.12)
        );

        border: 1px solid #6552a5;
        border-radius: 20px;
        padding: 26px 28px;
        margin: 0 0 24px 0;

        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.18);

        overflow-wrap: anywhere;
    }

    .hero-banner h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        line-height: 1.25;
        margin: 10px 0;
        text-shadow: none;
    }

    .hero-banner p {
        color: #e1eaff !important;
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .hero-tag {
        color: #67e8f9;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* ==========================================
       SIDEBAR
       ========================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #17132d,
            #101a2d
        );

        border-right: 1px solid #40386b;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #e7eaff !important;
    }

    /* ==========================================
       METRIC CARDS
       ========================================== */

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            #201b40,
            #14263d
        );

        border: 1px solid #514582;
        border-radius: 14px;
        padding: 16px;
        min-width: 0;
        overflow-wrap: anywhere;
    }

    div[data-testid="stMetricLabel"] {
        color: #c4b5fd !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        overflow-wrap: anywhere;
    }

    /* ==========================================
       TABS
       ========================================== */

    div[data-testid="stTabs"] button[role="tab"] {
        color: #e2d9ff !important;
        border-radius: 8px 8px 0 0;
        padding: 10px 12px;
        white-space: normal;
    }

    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: #67e8f9 !important;
        background: rgba(139, 92, 246, 0.14);
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #67e8f9 !important;
        border-bottom: 3px solid #a78bfa !important;
        background: rgba(139, 92, 246, 0.14);
    }

    /* ==========================================
       INPUT FIELDS
       White input backgrounds with dark text
       ========================================== */

    div[data-baseweb="input"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 9px !important;
    }

    div[data-baseweb="input"] input {
        background: #ffffff !important;
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        caret-color: #6d28d9 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }

    /* Number input +/- controls */
    div[data-testid="stNumberInput"] button {
        color: #ffffff !important;
        background: #28354a !important;
        border-color: #475569 !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background: #51418b !important;
        border-color: #a78bfa !important;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border-color: #cbd5e1 !important;
        border-radius: 9px !important;
    }

    div[data-baseweb="select"] * {
        color: #172033 !important;
    }

    /* Dropdown menu */
    div[role="listbox"],
    li[role="option"] {
        background: #ffffff !important;
        color: #172033 !important;
    }

    li[role="option"]:hover {
        background: #ede9fe !important;
    }

    /* Text areas */
    div[data-baseweb="textarea"] {
        background: #ffffff !important;
        border-radius: 9px !important;
    }

    div[data-baseweb="textarea"] textarea {
        background: #ffffff !important;
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        caret-color: #6d28d9 !important;
    }

    /* Radio buttons and checkboxes */
    div[data-testid="stRadio"] label,
    div[data-testid="stCheckbox"] label {
        color: #f1eaff !important;
    }

    /* ==========================================
       BUTTONS
       ========================================== */

    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stLinkButton"] a {
        background: linear-gradient(
            100deg,
            #7c3aed,
            #4f46e5,
            #087ea4
        ) !important;

        color: #ffffff !important;
        border: 1px solid #8171c7 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        min-height: 42px;
        transition: background 0.2s ease;
    }

    .stButton > button *,
    .stDownloadButton > button * {
        color: #ffffff !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(
            100deg,
            #6d28d9,
            #3730a3,
            #0e7490
        ) !important;

        border-color: #67e8f9 !important;
    }

    /* ==========================================
       FORM CONTAINERS
       ========================================== */

    div[data-testid="stForm"] {
        background: rgba(17, 24, 48, 0.90);
        border: 1px solid #514582;
        border-radius: 16px;
        padding: 20px;
    }

    /* ==========================================
       ALERTS
       Light message boxes always use dark text
       ========================================== */

    div[data-testid="stAlert"] {
        background: #edf4ff !important;
        border: 1px solid #b9c9e6 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] div {
        color: #1e293b !important;
    }

    /* ==========================================
       DATA TABLES
       ========================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #514582;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ==========================================
       PLOTLY CHARTS
       ========================================== */

    div[data-testid="stPlotlyChart"] {
        width: 100%;
        min-width: 0;
    }

    /* ==========================================
       FOOTER
       ========================================== */

    .anime-footer {
        text-align: center;
        color: #c4b5fd;
        font-size: 0.85rem;
        padding: 22px 0 5px 0;
        line-height: 1.7;
    }

    hr {
        border-color: #40386b !important;
    }

    /* ==========================================
       MOBILE RESPONSIVE STYLES
       ========================================== */

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.8rem 2rem 0.8rem;
        }

        .hero-banner {
            padding: 18px 16px;
            border-radius: 14px;
            margin-bottom: 18px;
        }

        .hero-banner h1 {
            font-size: 1.55rem;
            line-height: 1.3;
        }

        .hero-banner p {
            font-size: 0.92rem;
            line-height: 1.6;
        }

        .hero-tag {
            font-size: 0.65rem;
            letter-spacing: 1.4px;
        }

        h2 {
            font-size: 1.35rem !important;
        }

        h3 {
            font-size: 1.15rem !important;
        }

        div[data-testid="stForm"] {
            padding: 12px;
            border-radius: 12px;
        }

        div[data-testid="stMetric"] {
            padding: 10px;
            border-radius: 10px;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            padding: 8px 7px;
            font-size: 0.78rem;
        }

        .stButton > button,
        .stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] button {
            min-height: 44px;
            width: 100%;
        }

        div[data-testid="stPlotlyChart"] {
            overflow-x: auto;
        }
    }

    /* Extra-small phones */
    @media (max-width: 400px) {
        .block-container {
            padding-left: 0.55rem;
            padding-right: 0.55rem;
        }

        .hero-banner h1 {
            font-size: 1.35rem;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            font-size: 0.70rem;
            padding: 7px 4px;
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
    st.caption("⚡ Python · Streamlit · Random Forest")

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
        c2.metric("Rounded score", str(rounded_score))
        c3.metric("Estimated grade", grade)

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
                autosize=True,
                margin=dict(l=10, r=10, t=55, b=10),
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
                autosize=True,
                margin=dict(l=10, r=10, t=55, b=10),
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
            autosize=True,
            margin=dict(l=10, r=10, t=55, b=10),
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
            autosize=True,
            margin=dict(l=10, r=10, t=55, b=10),
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

        MAE and R² summarize model performance on held-out data.

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
