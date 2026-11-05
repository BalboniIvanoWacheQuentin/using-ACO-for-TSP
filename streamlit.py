
import streamlit as st

pg = st.navigation([
    st.Page("Fourmis.py", title="Démonstrateur", icon="🐜", default=True),
    st.Page("Analyse.py", title="Analyse", icon="📊"),
])

pg.run()