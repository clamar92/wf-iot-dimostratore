# app.py
import streamlit as st
st.set_page_config(page_title="DT Demo — Home", layout="wide")

st.markdown("""
<style>
header[data-testid="stHeader"] { display:none !important; }
.block-container { padding-top: .5rem; }
</style>
""", unsafe_allow_html=True)

st.title("DT Demo — Home")

if hasattr(st, "page_link"):
    st.page_link("pages/01_Step1.py", label="➡️ Apri Step 1")
    st.page_link("pages/02_Step2.py", label="➡️ Apri Step 2")
    st.page_link("pages/03_Step3.py", label="➡️ Apri Step 3")
    st.page_link("pages/04_Step4.py", label="➡️ Apri Step 4")
