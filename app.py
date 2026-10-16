from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "house_price_model.joblib"
DATA_FILE = BASE_DIR / "boston_housing.csv"

st.set_page_config(page_title="HomeValue | Price Estimator", page_icon="🏡", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem;}
.hero {padding: 1.6rem 1.8rem; border-radius: 18px; background: linear-gradient(120deg,#102a43,#1d4e89); color: white; margin-bottom: 1.2rem;}
.hero h1 {color:white; margin-bottom:.25rem; font-size:2.2rem;}
.hero p {color:#e5eef8; margin-bottom:0;}
.result {padding:1.25rem; border-radius:16px; background:#eef8f1; border:1px solid #b8e0c4; color:#153d25;}
.small-note {font-size:.9rem; color: #68778a;}
</style>
<div class="hero">
  <h1>🏡 HomeValue</h1>
  <p>Estimate a home's value using a machine-learning regression model.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_FILE.exists():
    st.error("The trained model file is missing. Run `python train_model.py` before starting the app.")
    st.stop()

bundle = joblib.load(MODEL_FILE)
model = bundle["model"]

with st.sidebar:
    st.header("About this demo")
    st.write("Enter property/neighborhood characteristics and select **Estimate price**.")
    st.caption("This model was trained on the supplied Boston Housing dataset. It is an educational demo, not a live real-estate valuation service.")
    st.divider()
    st.write("**Target units:** MEDV is measured in thousands of US dollars in this historical dataset.")

st.subheader("Property and neighborhood details")
st.write("Use the dataset's feature definitions and units. Defaults are typical example values, not live market data.")

with st.form("prediction_form"):
    left, mid, right = st.columns(3)
    with left:
        crim = st.number_input("CRIM · Crime rate", min_value=0.0, value=0.05, step=0.01, help="Per-capita crime rate by town.")
        zn = st.number_input("ZN · Residential land (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
        indus = st.number_input("INDUS · Non-retail business acres (%)", min_value=0.0, value=8.0, step=0.5)
        chas = st.selectbox("CHAS · Charles River tract?", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)")
        nox = st.number_input("NOX · Nitric oxides concentration", min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.3f")
    with mid:
        rm = st.number_input("RM · Average rooms per dwelling", min_value=1.0, max_value=15.0, value=6.0, step=0.1)
        age = st.number_input("AGE · Older owner-occupied units (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
        dis = st.number_input("DIS · Distance to employment centres", min_value=0.0, value=4.0, step=0.1)
        rad = st.number_input("RAD · Highway accessibility index", min_value=1, max_value=24, value=4, step=1)
    with right:
        tax = st.number_input("TAX · Property tax rate", min_value=0.0, value=300.0, step=10.0)
        ptratio = st.number_input("PTRATIO · Pupil-teacher ratio", min_value=1.0, max_value=40.0, value=18.0, step=0.5)
        b_value = st.number_input("B · Dataset demographic index", min_value=0.0, value=390.0, step=1.0, help="Historical dataset variable; included to match the original model features.")
        lstat = st.number_input("LSTAT · Lower-status population (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.5)
    submitted = st.form_submit_button("✨ Estimate price", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "CRIM": crim, "ZN": zn, "INDUS": indus, "CHAS": chas, "NOX": nox,
        "RM": rm, "AGE": age, "DIS": dis, "RAD": rad, "TAX": tax,
        "PTRATIO": ptratio, "B": b_value, "LSTAT": lstat,
    }], columns=bundle["features"])
    prediction = float(model.predict(row)[0])
    st.markdown("### Your estimate")
    st.markdown(f'<div class="result"><div style="font-size:.95rem">Predicted value (dataset units)</div><div style="font-size:2.4rem;font-weight:750">${prediction * 1000:,.0f}</div><div>Equivalent to ${prediction:,.2f} thousand in the dataset target scale.</div></div>', unsafe_allow_html=True)
    st.info("Important: this is a prediction from a historical Boston-area teaching dataset. It does not use a live property listing, address, current market prices, inflation adjustment, or today's local conditions. Do not use it for a real purchase or sale decision.")
    with st.expander("See the values sent to the model"):
        st.dataframe(row, use_container_width=True, hide_index=True)

st.divider()
st.markdown('<p class="small-note">Built with Python, Pandas, scikit-learn and Streamlit · Educational machine-learning project</p>', unsafe_allow_html=True)
