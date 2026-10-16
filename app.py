from pathlib import Path
import math

import joblib
import pandas as pd
import streamlit as st
import textwrap


# ============================================================
# HOMEVALUE | INDIAN PROPERTY INTELLIGENCE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "house_price_model.joblib"
DATA_FILE = BASE_DIR / "indian.csv"
DICTIONARY_FILE = BASE_DIR / "indian_data_dictionary.csv"

TARGET = "MEDV_LAKH"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HomeValue | Property Intelligence",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PREMIUM UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --background: #f6f5ef;
        --surface: #ffffff;
        --primary: #234735;
        --primary-light: #32654b;
        --gold: #c6a56b;
        --text: #1c3026;
        --muted: #718078;
        --border: #e5e8df;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 90% 0%,
                rgba(198,165,107,0.12),
                transparent 28rem
            ),
            var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: #182d23;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f6f4ec;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #d0d9d1;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        color: var(--primary);
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.19em;
        margin-bottom: 1.2rem;
    }

    .brand-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 13px;
        background: var(--primary);
        color: white;
        font-size: 1.45rem;
        box-shadow: 0 8px 20px rgba(35,71,53,0.18);
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.5rem;
        border-radius: 26px;
        background: linear-gradient(
            120deg,
            #172e23 0%,
            #234735 60%,
            #3a6248 100%
        );
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 18px 45px rgba(29,54,42,0.12);
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        right: -65px;
        top: -115px;
        border: 1px solid rgba(230,205,155,0.38);
        border-radius: 50%;
        box-shadow:
            0 0 0 30px rgba(230,205,155,0.04),
            0 0 0 65px rgba(230,205,155,0.035);
    }

    .hero-label {
        color: #e3c894;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero h1 {
        color: #ffffff;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.2rem, 4vw, 3.5rem);
        font-weight: 800;
        letter-spacing: -0.055em;
        line-height: 1.12;
        margin: 0 0 0.8rem;
    }

    .hero p {
        color: #dce7de;
        font-size: 1rem;
        line-height: 1.8;
        max-width: 650px;
        margin-bottom: 0;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 1.3rem;
        padding: 0.5rem 0.9rem;
        border: 1px solid rgba(255,255,255,0.22);
        border-radius: 999px;
        color: #f3e8d2;
        font-size: 0.78rem;
    }

    .section-label {
        color: #9b7b46;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.17em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .section-heading {
        color: #1c3026;
        font-family: 'Manrope', sans-serif;
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.35rem;
    }

    .section-description {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.7;
        margin-bottom: 1.4rem;
    }

    .result-card {
        padding: 1.8rem;
        border-radius: 22px;
        background: linear-gradient(125deg, #ffffff, #eff5ee);
        border: 1px solid #dce7dc;
        box-shadow: 0 12px 32px rgba(30,65,45,0.07);
        margin: 0.6rem 0 1.2rem;
    }

    .result-label {
        color: #6e7c72;
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        text-transform: uppercase;
    }

    .result-price {
        color: #214b36;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.3rem, 5vw, 3.5rem);
        font-weight: 800;
        letter-spacing: -0.06em;
        line-height: 1.3;
        margin: 0.55rem 0;
        overflow-wrap: anywhere;
    }

    .result-note {
        color: #6e7c72;
        font-size: 0.87rem;
        line-height: 1.7;
    }

    .metric-card {
        min-height: 105px;
        padding: 1.1rem;
        border: 1px solid var(--border);
        border-radius: 17px;
        background: white;
    }

    .metric-label {
        color: #78847b;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .metric-value {
        color: #234735;
        font-family: 'Manrope', sans-serif;
        font-size: 1.3rem;
        font-weight: 800;
        margin-top: 0.45rem;
    }

    div[data-testid="stFormSubmitButton"] > button {
        background: #234735;
        color: white;
        border: 1px solid #234735;
        border-radius: 12px;
        min-height: 3.2rem;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #32654b;
        border-color: #32654b;
        color: white;
        box-shadow: 0 7px 18px rgba(35,71,53,0.15);
    }

    div[data-testid="stNumberInput"] input {
        border-radius: 10px;
    }

    .footer {
        border-top: 1px solid #e1e5dc;
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        color: #788279;
        font-size: 0.79rem;
        line-height: 1.8;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 1.5rem;
            border-radius: 20px;
        }

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FEATURE DEFINITIONS
# Based on indian_data_dictionary.csv
# ============================================================

FEATURE_INFO = {
    "CRIME_RATE": {
        "label": "Crime Rate",
        "description": "Cognizable crimes per lakh population.",
    },
    "PLOTTED_ZONE_PCT": {
        "label": "Plotted Residential Zone",
        "description": "Residential land zoned for large plots (%).",
    },
    "INDUS_PCT": {
        "label": "Industrial Land",
        "description": "Non-retail business and industrial land (%).",
    },
    "WATERFRONT": {
        "label": "Waterfront Property",
        "description": "Whether the area borders a river, lake, or sea.",
    },
    "NO2": {
        "label": "NO₂ Concentration",
        "description": "Annual mean nitrogen dioxide concentration (µg/m³).",
    },
    "ROOMS": {
        "label": "Average Rooms",
        "description": "Average rooms per dwelling, excluding kitchen and bathroom.",
    },
    "AGE_30PLUS_PCT": {
        "label": "Older Housing",
        "description": "Dwellings more than 30 years old (%).",
    },
    "DIST_HUBS_KM": {
        "label": "Distance to Employment Hubs",
        "description": "Weighted distance to major employment hubs (km).",
    },
    "TRANSIT_INDEX": {
        "label": "Transit Accessibility",
        "description": "Accessibility index for metro, rail, and major roads.",
    },
    "PROP_TAX_PER_10L": {
        "label": "Property Tax",
        "description": "Annual property tax per ₹10 lakh of property value.",
    },
    "PTR": {
        "label": "Pupil–Teacher Ratio",
        "description": "Students per teacher in local schools.",
    },
    "SLUM_POOR_PCT": {
        "label": "Informal Housing / Deprivation",
        "description": "Households in informal housing or deprived conditions (%).",
    },
    "FEMALE_LITERACY_PCT": {
        "label": "Female Literacy Rate",
        "description": "Female literacy rate (%).",
    },
}


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

if not DATA_FILE.exists():
    st.error("Dataset file `indian.csv` was not found.")
    st.info("Place indian.csv in the same folder as app.py.")
    st.stop()

try:
    data = pd.read_csv(DATA_FILE)
except Exception as exc:
    st.error("The dataset could not be loaded.")
    st.exception(exc)
    st.stop()


if TARGET not in data.columns:
    st.error(
        f"The dataset must contain the target column `{TARGET}`. "
        f"Available columns: {', '.join(data.columns)}"
    )
    st.stop()


if not MODEL_FILE.exists():
    st.error("The trained model file is missing.")
    st.info("Run `python train_model.py` to create the model.")
    st.stop()


try:
    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]
    features = list(bundle["features"])
    saved_target = bundle.get("target", TARGET)
except Exception as exc:
    st.error("The saved model could not be loaded.")
    st.exception(exc)
    st.stop()


# Confirm the saved model and dataset agree.
missing_features = [
    feature for feature in features
    if feature not in data.columns
]

if missing_features:
    st.error("The saved model and dataset do not match.")
    st.write("Missing columns:", missing_features)
    st.info(
        "Check train_model.py, then retrain the model using the current dataset."
    )
    st.stop()


if saved_target != TARGET:
    st.error(
        f"The saved model target is `{saved_target}`, "
        f"but this application expects `{TARGET}`."
    )
    st.info("Update train_model.py and retrain the model.")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            color:#d7bd88;
            font-size:.75rem;
            font-weight:800;
            letter-spacing:.2em;
        ">
            HOMEVALUE
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## Property Intelligence")

    st.write(
        "Explore an estimated home value using a trained "
        "machine-learning regression model."
    )

    st.divider()

    st.markdown("### Model overview")

    st.caption("Algorithm: Random Forest Regression")
    st.caption(f"Input features: {len(features)}")
    st.caption(f"Dataset records: {len(data):,}")
    st.caption("Target: Median home value (INR lakh)")

    st.divider()

    st.markdown("### Responsible use")

    st.caption(
        "This is an educational model. It is not a verified "
        "property appraisal and should not be used as the sole "
        "basis for a purchase, sale, loan, or investment decision."
    )
# ============================================================
# HERO SECTION
# ============================================================

st.html("""
<div class="brand">
    <span class="brand-icon">⌂</span>
    HOMEVALUE
    <span style="color:#a48650;">/</span>
    PROPERTY INTELLIGENCE
</div>

<div class="hero">
    <div class="hero-label">
        A smarter way to explore property values
    </div>

    <h1>
        Discover your property's<br>
        estimated value.
    </h1>

    <p>
        Explore an estimated property value using a trained
        machine-learning model and the property characteristics
        available in your dataset.
    </p>

    <span class="hero-badge">
        Machine Learning &nbsp;·&nbsp; Property Analytics
    </span>
</div>
""")

# ============================================================
# PROPERTY INPUT FORM
# ============================================================

st.markdown(
    '<div class="section-label">01 / Property profile</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-heading">Tell us about the property</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
        Complete the property and neighbourhood details below.
        Input ranges are based on the supplied dataset.
    </div>
    """,
    unsafe_allow_html=True,
)


# Use medians and observed ranges to generate defaults.
# The feature order comes from the trained model bundle.

feature_specs = []

for feature in features:

    numeric = pd.to_numeric(data[feature], errors="coerce")
    clean = numeric.dropna()

    if clean.empty:
        st.error(f"No valid numeric values found for `{feature}`.")
        st.stop()

    minimum = float(clean.min())
    maximum = float(clean.max())
    default = float(clean.median())

    integer_feature = bool(
        (clean % 1 == 0).all()
    )

    if integer_feature:
        step = 1.0
    elif abs(default) >= 100:
        step = 1.0
    elif abs(default) >= 10:
        step = 0.1
    else:
        step = 0.01

    feature_specs.append(
        {
            "name": feature,
            "min": minimum,
            "max": maximum,
            "default": default,
            "integer": integer_feature,
            "step": step,
        }
    )


values = {}

with st.form("homevalue_prediction_form"):

    columns = st.columns(3, gap="large")

    for index, spec in enumerate(feature_specs):

        feature = spec["name"]
        info = FEATURE_INFO.get(
            feature,
            {
                "label": feature.replace("_", " ").title(),
                "description": "Enter a value within the observed dataset range.",
            },
        )

        with columns[index % 3]:

            st.markdown(
                f"**{index + 1:02d} · {info['label']}**"
            )

            st.caption(info["description"])

            minimum = spec["min"]
            maximum = spec["max"]
            default = spec["default"]

            if math.isclose(minimum, maximum):

                # A constant column needs a small widget range.
                widget_min = minimum - 1
                widget_max = maximum + 1

                value = st.number_input(
                    info["label"],
                    min_value=widget_min,
                    max_value=widget_max,
                    value=default,
                    step=spec["step"],
                    key=f"feature_{feature}",
                    help=f"Observed dataset value: {minimum:g}.",
                )

            elif feature == "WATERFRONT":

                value = st.selectbox(
                    info["label"],
                    options=[0, 1],
                    index=int(round(default)),
                    format_func=lambda x: (
                        "No" if x == 0 else "Yes"
                    ),
                    key=f"feature_{feature}",
                    help=info["description"],
                )

            elif spec["integer"]:

                low = math.floor(minimum)
                high = math.ceil(maximum)

                value = st.number_input(
                    info["label"],
                    min_value=low,
                    max_value=high,
                    value=min(max(round(default), low), high),
                    step=1,
                    key=f"feature_{feature}",
                    help=f"Observed range: {minimum:g} to {maximum:g}.",
                )

            else:

                value = st.number_input(
                    info["label"],
                    min_value=minimum,
                    max_value=maximum,
                    value=min(max(default, minimum), maximum),
                    step=spec["step"],
                    format="%.2f",
                    key=f"feature_{feature}",
                    help=f"Observed range: {minimum:g} to {maximum:g}.",
                )

            values[feature] = value

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "✦  Estimate Property Value",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# PREDICTION RESULT
# ============================================================

if submitted:

    try:

        # Preserve the exact feature order used during training.
        row = pd.DataFrame(
            [
                {
                    feature: values[feature]
                    for feature in features
                }
            ],
            columns=features,
        )

        prediction = float(model.predict(row)[0])

        if not math.isfinite(prediction):
            raise ValueError(
                "The model returned an invalid prediction."
            )

        if prediction < 0:
            st.warning(
                "The model returned a negative value. "
                "Review the model and dataset before interpreting this result."
            )

        # MEDV_LAKH is measured in INR lakh.
        price_lakh = prediction
        price_crore = prediction / 100

        st.markdown("---")

        st.markdown(
            '<div class="section-label">02 / Valuation result</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-heading">Your estimate is ready</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-label">
                    Estimated property value
                </div>

                <div class="result-price">
                    ₹{price_lakh:,.2f} lakh
                </div>

                <div class="result-note">
                    Approximately ₹{price_crore:,.2f} crore.
                    <br>
                    This is a machine-learning prediction,
                    not a confirmed market valuation.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_columns = st.columns(3)

        with metric_columns[0]:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Input features</div>
                    <div class="metric-value">{len(features)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric_columns[1]:

            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">ML algorithm</div>
                    <div class="metric-value">Random Forest</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with metric_columns[2]:

            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">Prediction status</div>
                    <div class="metric-value">Complete</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.info(
            "Important: the result is generated from the supplied "
            "dataset and its trained model. It does not verify a property "
            "address, current listings, current market conditions, "
            "property ownership, or legal details. Do not rely on it "
            "alone for a real purchase, sale, or investment decision."
        )

        with st.expander("Review the values submitted to the model"):

            st.dataframe(
                row,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                label="Download input values as CSV",
                data=row.to_csv(index=False).encode("utf-8"),
                file_name="homevalue_prediction_inputs.csv",
                mime="text/csv",
            )

    except Exception as exc:

        st.error(
            "Prediction failed. Check that the model, dataset, "
            "feature columns, and library versions are compatible."
        )

        st.exception(exc)


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">

    <strong style="color:#234735;">
        HOMEVALUE
    </strong>

    &nbsp; · &nbsp; Built with Python, Pandas,
    scikit-learn and Streamlit.

    <br><br>

    An educational machine-learning project.
    Predictions are estimates, not professional valuations.

</div>
""")