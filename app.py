from pathlib import Path

import joblib
import numpy as np
import streamlit as st


MODEL_PATH = Path(__file__).with_name("iris_model.pkl")
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Playfair+Display:wght@600;700&display=swap');
    .stApp { background: #f4f0e8; color: #20352d; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { display: none; }
    .block-container { max-width: 1080px; padding: 3rem 2rem 4rem; }
    h1, h2, h3 { font-family: 'Playfair Display', Georgia, serif; }
    p, label, .stButton, [data-testid="stMetricValue"] { font-family: 'DM Sans', sans-serif; }
    .hero { background: #20352d; border-radius: 20px; padding: 2.4rem 2.7rem; margin-bottom: 1.5rem; position: relative; overflow: hidden; }
    .hero::after { content: '✽'; color: #e4ad55; font-size: 11rem; line-height: 1; position: absolute; right: 2rem; top: -2.4rem; opacity: .22; }
    .eyebrow { color: #e4ad55; font: 700 .75rem 'DM Sans', sans-serif; letter-spacing: .16em; text-transform: uppercase; margin: 0 0 .7rem; }
    .hero h1 { color: #fffaf1; font-size: clamp(2.2rem, 5vw, 4rem); margin: 0; position: relative; z-index: 1; }
    .hero p { color: #c9d3c5; font-size: 1.05rem; max-width: 38rem; margin: .75rem 0 0; position: relative; z-index: 1; }
    .snapshot { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin: 0 0 1.5rem; }
    .snapshot-item { border-radius: 14px; padding: 1rem 1.1rem; color: #20352d; }
    .snapshot-item:nth-child(1) { background: #f1d8bd; }
    .snapshot-item:nth-child(2) { background: #d4e6d3; }
    .snapshot-item:nth-child(3) { background: #e8cfe0; }
    .snapshot-number { color: #20352d; font: 700 1.7rem 'Playfair Display', Georgia, serif; }
    .snapshot-label { color: #53685d; font: 500 .78rem 'DM Sans', sans-serif; margin-top: .15rem; }
    .panel { background: #fffaf1; border: 1px solid #ded8ca; border-radius: 16px; padding: 1.5rem 1.6rem .8rem; min-height: 100%; }
    .panel h2 { color: #20352d; font-size: 1.55rem; margin: 0 0 .35rem; }
    .panel-copy { color: #718078; margin: 0 0 1.2rem; font-size: .92rem; }
    div[data-testid="stSlider"] label { color: #456054; font-weight: 500; }
    .readout { background: #e8efe5; border-radius: 12px; padding: 1.3rem; margin: .5rem 0 1rem; }
    .readout-label { color: #718078; font: 700 .72rem 'DM Sans', sans-serif; letter-spacing: .1em; text-transform: uppercase; }
    .readout-title { color: #20352d; font: 700 1.55rem 'Playfair Display', Georgia, serif; margin-top: .3rem; }
    .readout-copy { color: #53685d; font: .88rem 'DM Sans', sans-serif; margin-top: .35rem; }
    .guide-title { color: #20352d; font: 700 1.7rem 'Playfair Display', Georgia, serif; margin: 2rem 0 .8rem; }
    .guide { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; }
    .species { border-radius: 14px; padding: 1.1rem 1.2rem; min-height: 9rem; }
    .species h3 { margin: 0 0 .35rem; font-size: 1.25rem; }
    .species p { color: #53685d; font: .82rem/1.45 'DM Sans', sans-serif; margin: 0; }
    .setosa { background: #d9e8f0; border-top: 5px solid #4e8ca8; }
    .versicolor { background: #f2e2b9; border-top: 5px solid #d29a36; }
    .virginica { background: #ead6e4; border-top: 5px solid #a46191; }
    .stButton > button { background: #c27a58; border: 0; border-radius: 10px; color: #fffaf1; font-weight: 700; min-height: 2.8rem; }
    .stButton > button:hover { background: #a96143; color: #fffaf1; }
    @media (max-width: 700px) { .snapshot, .guide { grid-template-columns: 1fr; } .hero { padding: 2rem 1.5rem; } }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="snapshot">
        <div class="snapshot-item"><div class="snapshot-number">150</div><div class="snapshot-label">training samples</div></div>
        <div class="snapshot-item"><div class="snapshot-number">3</div><div class="snapshot-label">iris species</div></div>
        <div class="snapshot-item"><div class="snapshot-number">4</div><div class="snapshot-label">measurements analyzed</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <p class="eyebrow">Botanical intelligence · Iris study</p>
        <h1>Find the flower<br>in the details.</h1>
        <p>Measure four simple features and let the trained model identify the iris species.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

controls, preview = st.columns([1.25, .75], gap="large")

with controls:
    st.markdown('<div class="panel"><h2>Flower measurements</h2><p class="panel-copy">Adjust the sliders to match your specimen.</p>', unsafe_allow_html=True)
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.2, 0.1)
    st.markdown('</div>', unsafe_allow_html=True)

with preview:
    st.markdown('<div class="panel"><h2>Specimen profile</h2><p class="panel-copy">Current measurements in centimeters.</p>', unsafe_allow_html=True)
    st.metric("Sepal length", f"{sepal_length:.1f} cm")
    st.metric("Sepal width", f"{sepal_width:.1f} cm")
    st.metric("Petal length", f"{petal_length:.1f} cm")
    st.metric("Petal width", f"{petal_width:.1f} cm")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
if st.button("Predict species", type="primary", use_container_width=True):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    st.markdown(
        f'<div class="readout"><div class="readout-label">Model result · K-nearest neighbors</div><div class="readout-title">{prediction}</div><div class="readout-copy">Matched using sepal length, sepal width, petal length, and petal width.</div></div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="guide-title">Meet the three species</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="guide">
        <div class="species setosa"><h3>Setosa</h3><p>Usually the smallest petals, with a compact and distinctive profile.</p></div>
        <div class="species versicolor"><h3>Versicolor</h3><p>A middle-sized profile that often bridges the other two varieties.</p></div>
        <div class="species virginica"><h3>Virginica</h3><p>Often the largest petals, creating a broader overall measurement pattern.</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)
