# pages/02_Empty.py
import streamlit as st
from dataclasses import dataclass, field

st.set_page_config(page_title="DT Request Demo — Step 4", layout="wide")

# CSS: nascondi header e no scrollbar orizzontale
st.markdown("""
<style>
header[data-testid="stHeader"] { display: none !important; }
.block-container { padding-top: 0.5rem; }
html, body, [data-testid="stAppViewContainer"] { overflow-x: hidden !important; }
h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { pointer-events: none !important; text-decoration: none !important; }
h1 a svg, h2 a svg, h3 a svg, h4 a svg, h5 a svg, h6 a svg { display: none !important; }
</style>
""", unsafe_allow_html=True)

# Se in futuro vuoi uno state ANCHE qui, usa un'altra chiave:
SCENE_KEY = "scene_step4"
if SCENE_KEY not in st.session_state:
    st.session_state[SCENE_KEY] = dict()   # placeholder semplice

st.title("Step 4 (vuoto)")

# link di ritorno (opzionale, se disponibile nella tua versione di Streamlit)
if hasattr(st, "page_link"):
    st.page_link("app.py", label="⬅️ Torna a Step 1")

# Per ora non fai nulla
st.write("")
