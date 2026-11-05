import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PyPDF2 import PdfReader, PdfWriter

st.set_page_config(page_title="Présentation", page_icon="📑", layout="wide")
st.title("📑")

PDF_PATH = Path("presentation.pdf")

if not PDF_PATH.exists():
    st.error("Fichier presentation.pdf introuvable.")
    st.stop()


with open("presentation.pdf", "rb") as f:
    pdf_bytes = f.read()

pdf_b64 = base64.b64encode(pdf_bytes).decode()

components.html(
    f"""
    <iframe
        src="data:application/pdf;base64,{pdf_b64}"
        width="100%"
        height="800">
    </iframe>
    """,
    height=820,
)

with open("presentation.pdf", "rb") as pdf:
    st.download_button(
        "Télécharger le PDF",
        pdf,
        "presentation.pdf"
    )

