"""Streamlit app: predict diamond price from 9 attributes using the model trained in diamond_price_model.ipynb."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "diamond_model.joblib"
META_PATH = BASE / "models" / "model_metadata.json"

st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        st.error("Model not found. Run `diamond_price_model.ipynb` first to create `models/diamond_model.joblib`.")
        st.stop()
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


model, meta = load_artifacts()
ranges = meta["numeric_ranges"]
order = meta["category_order"]


def num_input(col, label, default, step, fmt, help_text=None):
    r = ranges[col]
    return st.number_input(label, min_value=r["min"], max_value=r["max"], value=default,
                           step=step, format=fmt, help=help_text)


st.title("💎 Diamond Price Predictor")
st.caption(f"Model: **{meta['model_name']}** · test R² {meta['test_metrics']['r2']:.3f} · "
           f"typical error ±${meta['test_metrics']['mae']:,.0f} · trained on {meta['n_training_rows']:,} diamonds")

with st.form("diamond"):
    st.subheader("Grades")
    g1, g2, g3, g4 = st.columns(4)
    carat = g1.number_input("Carat", min_value=ranges["carat"]["min"], max_value=ranges["carat"]["max"],
                            value=1.0, step=0.01, format="%.2f", help="Weight of the diamond")
    cut = g2.selectbox("Cut", order["cut"][::-1], help="Best → worst")
    color = g3.selectbox("Color", order["color"][::-1], index=3, help="D (colourless, best) → J")
    clarity = g4.selectbox("Clarity", order["clarity"][::-1], index=3, help="IF (flawless, best) → I1")

    st.subheader("Proportions")
    p1, p2 = st.columns(2)
    depth = p1.number_input("Depth %", min_value=ranges["depth"]["min"], max_value=ranges["depth"]["max"],
                            value=61.8, step=0.1, format="%.1f", help="Total depth % = z / mean(x, y) × 100")
    table = p2.number_input("Table %", min_value=ranges["table"]["min"], max_value=ranges["table"]["max"],
                            value=57.0, step=0.5, format="%.1f", help="Width of top facet relative to widest point")

    st.subheader("Dimensions (mm)")
    d1, d2, d3 = st.columns(3)
    with d1:
        x = num_input("x", "Length (x)", 6.40, 0.01, "%.2f")
    with d2:
        y = num_input("y", "Width (y)", 6.40, 0.01, "%.2f")
    with d3:
        z = num_input("z", "Depth (z)", 3.95, 0.01, "%.2f")
    st.caption("Tip: a 1.00 ct round ideal-cut diamond is roughly 6.4 × 6.4 × 3.95 mm.")

    submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{"carat": carat, "cut": cut, "color": color, "clarity": clarity,
                         "depth": depth, "table": table, "x": x, "y": y, "z": z}])[meta["features"]]
    price = float(model.predict(row)[0])
    mape = meta["test_metrics"]["mape"]

    st.metric("Estimated price", f"${price:,.0f}")
    st.caption(f"Typical range: ${price * (1 - mape):,.0f} – ${price * (1 + mape):,.0f} (± average % error on test set)")

    # Consistency checks: size inputs should agree with each other
    warnings = []
    density = carat / (x * y * z)
    lo, hi = meta["carat_per_mm3"]["p01"] * 0.9, meta["carat_per_mm3"]["p99"] * 1.1
    if not lo <= density <= hi:
        expected = carat / ((lo + hi) / 2)
        warnings.append(f"Carat and x·y·z don't match typical diamonds (x·y·z = {x * y * z:.1f} mm³; "
                        f"{carat:.2f} ct usually ≈ {expected:.1f} mm³). Check your dimensions.")
    calc_depth = z / ((x + y) / 2) * 100
    if abs(calc_depth - depth) > 2:
        warnings.append(f"Depth % entered ({depth:.1f}) differs from z / mean(x, y) = {calc_depth:.1f}.")
    for w in warnings:
        st.warning(w)
    if warnings:
        st.caption("The prediction may be unreliable when inputs are inconsistent.")

    with st.expander("Input sent to the model"):
        st.dataframe(row, hide_index=True)

with st.expander("Model comparison (5-fold CV)"):
    st.dataframe(pd.DataFrame(meta["cv_results"]), hide_index=True)
