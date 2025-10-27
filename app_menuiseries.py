# app_menuiseries.py
# Streamlit app: Extraction ciblée (repère, quantité, largeur, hauteur, finition) depuis des devis PDF
# Usage: streamlit run app_menuiseries.py
# Dépendances: streamlit, pdfplumber, pandas, pillow, pytesseract (si OCR activé)

import io
import os
import re
import zipfile
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pdfplumber
import pandas as pd

# OCR optionnel
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

st.set_page_config(page_title="Extraction menuiseries", page_icon="🪟", layout="wide")

# ----------- Utils -----------

X_CHARS = r"[xX×\*]"  # séparateur de dimensions
NUM = r"\d{2,4}(?:[ \u00A0]\d{3})?(?:[.,]\d+)?"

DIMENSION_RE = re.compile(
    rf'(?P<w>{NUM})\s*{X_CHARS}\s*(?P<h>{NUM})(?:\s*(?P<unit>mm|cm|m))?'
)

# variations: H x L, Lxh, etc. (capturer dans n'importe quel ordre si taggé)
DIM_TAGGED_RE = re.compile(
    rf'(?:L(?:arg)?\s*[:=]?\s*(?P<w>{NUM})\s*(?P<unitw>mm|cm|m)?)'
    rf'.{{0,20}}?'
    rf'(?:H(?:aut)?\s*[:=]?\s*(?P<h>{NUM})\s*(?P<unith>mm|cm|m)?)',
    re.I
)

QTE_PATTERNS = [
    re.compile(r'(?:Qt[eé]|Quantit[eé])\s*[:\-]?\s*(?P<q>\d+)', re.I),
    re.compile(r'(?P<q>\d+)\s*(?:u|unit[eé]s?|pcs?|pi[eè]ces?)\b', re.I),
    re.compile(r'^[\s>]*?(?P<q>\d{1,3})\s*[xX]\b'),  # "2 x Fenêtre ..."
]

REPERE_PATTERNS = [
    re.compile(r'\bRep(?:[èe]re|\.?)\s*[:\-]?\s*(?P<r>[A-Za-z0-9\-_/]+)', re.I),
    re.compile(r'\bR(?:ep)?\s*[:\-]?\s*(?P<r>[A-Za-z0-9]{1,10})\b', re.I),
    re.compile(r'\b(?P<r>[A-Z]{1,4}\d{1,4}(?:-[A-Z0-9]{1,4})?)\b'),
]

FINITIONS = [
    r'RAL\s*\d{3,4}', r'BLANC', r'BLANC\s+RAL\s*\d{3,4}', r'GRIS',
    r'NOIR', r'ANODIS[ÉE]?', r'LAQU[ÉE]?', r'NATUREL', r'CH[ÊE]NE', r'ACAJOU',
    r'GRIS\s+ANTHRACITE', r'7016', r'9005', r'9016', r'7022'
]
FINIT_RE = re.compile("|".join(FINITIONS), re.I)

def to_float(num_str: Optional[str]) -> Optional[float]:
    if not num_str:
        return None
    s = re.sub(r"\s+", "", num_str)
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def to_mm(value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if not unit:
        return value  # suppose déjà mm
    u = unit.lower()
    if u == "mm":
        return value
    if u == "cm":
        return value * 10.0
    if u == "m":
        return value * 1000.0
    return value

def extract_text_from_pdf(file_bytes: bytes, do_ocr: bool, ocr_lang: str = "fra") -> List[str]:
    """
    Retourne une liste de textes par page. OCR si nécessaire (requiert Tesseract installé).
    """
    texts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if (not t.strip()) and do_ocr and OCR_AVAILABLE:
                pil = page.to_image(resolution=220).original.convert("RGB")
                t = pytesseract.image_to_string(pil, lang=ocr_lang)
            texts.append(t or "")
    return texts

def parse_line_for_fields(line: str) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[str], Optional[str]]:
    """
    Retourne (largeur_mm, hauteur_mm, quantite, repere, finition) trouvés sur la ligne.
    """
    largeur_mm = hauteur_mm = None
    quantite = None
    repere = None
    finition = None

    # Dimensions - format "L x H"
    m = DIMENSION_RE.search(line)
    if m:
        w = to_float(m.group("w"))
        h = to_float(m.group("h"))
        unit = m.group("unit")
        largeur_mm = to_mm(w, unit)
        hauteur_mm = to_mm(h, unit)
    else:
        # Format taggé "L: 1200 mm H: 1350 mm"
        m2 = DIM_TAGGED_RE.search(line)
        if m2:
            w = to_float(m2.group("w"))
            h = to_float(m2.group("h"))
            uw = m2.group("unitw")
            uh = m2.group("unith")
            largeur_mm = to_mm(w, uw)
            hauteur_mm = to_mm(h, uh)

    # Quantité
    for pat in QTE_PATTERNS:
        mq = pat.search(line)
        if mq:
            try:
                quantite = int(mq.group("q"))
            except Exception:
                pass
            if quantite is not None:
                break

    # Repère
    for pat in REPERE_PATTERNS:
        mr = pat.search(line)
        if mr:
            repere = mr.group("r")
            break

    # Finition
    mf = FINIT_RE.search(line)
    if mf:
        finition = mf.group(0)

    return largeur_mm, hauteur_mm, quantite, repere, finition

def parse_text(text: str, source: str) -> List[Dict[str, Any]]:
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or len(line) < 4:
            continue

        largeur_mm, hauteur_mm, quantite, repere, finition = parse_line_for_fields(line)
        # Ne garder que les lignes avec dimensions trouvées
        if largeur_mm is None or hauteur_mm is None:
            continue

        rows.append({
            "fichier": os.path.basename(source),
            "ligne_source": line,
            "repere": repere,
            "quantite": quantite,
            "largeur_mm": largeur_mm,
            "hauteur_mm": hauteur_mm,
            "finition": finition,
        })
    return rows

# ----------- UI -----------

st.title("🪟 Extraction menuiseries (repère, Qté, L, H, finition)")
st.caption("Chargez des PDF (ou un .zip de plusieurs PDF). L’app extrait les dimensions et métadonnées utiles.")

with st.sidebar:
    st.header("Paramètres")
    allow_zip = st.toggle("Autoriser fichiers ZIP (lot de PDF)", value=True)
    do_ocr = st.toggle("Activer OCR fallback (Tesseract requis)", value=False)
    ocr_lang = st.selectbox("Langue OCR", ["fra", "eng", "deu"], index=0)
    show_text = st.toggle("Montrer le texte brut (debug)", value=False)
    st.markdown("---")
    st.caption("Astuce: assurez-vous que les dimensions apparaissent sous forme '1200 x 1350' ou 'L:1200 H:1350'.")

accept_types = ["pdf"]
if allow_zip:
    accept_types.append("zip")

uploads = st.file_uploader("Déposez un ou plusieurs fichiers", type=accept_types, accept_multiple_files=True)

if uploads:
    all_rows: List[Dict[str, Any]] = []
    debug_texts = []

    for up in uploads:
        name = up.name.lower()
        data = up.read()

        if name.endswith(".zip"):
            # Dézipper en mémoire
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    for info in zf.infolist():
                        if info.filename.lower().endswith(".pdf"):
                            pdf_bytes = zf.read(info)
                            # Texte par page
                            texts = extract_text_from_pdf(pdf_bytes, do_ocr=do_ocr, ocr_lang=ocr_lang)
                            full = "\n".join(texts)
                            rows = parse_text(full, info.filename)
                            all_rows.extend(rows)
                            if show_text:
                                debug_texts.append((info.filename, full[:4000]))
            except zipfile.BadZipFile:
                st.error(f"Archive ZIP invalide: {up.name}")
        elif name.endswith(".pdf"):
            texts = extract_text_from_pdf(data, do_ocr=do_ocr, ocr_lang=ocr_lang)
            full = "\n".join(texts)
            rows = parse_text(full, up.name)
            all_rows.extend(rows)
            if show_text:
                debug_texts.append((up.name, full[:4000]))

    if not all_rows:
        st.warning("Aucune ligne avec dimensions détectées. Essayez d'activer l'OCR, ou partagez un exemple de ligne pour ajuster les motifs.")
    else:
        df = pd.DataFrame(all_rows)
        st.success(f"{len(df)} ligne(s) détectée(s). Vous pouvez corriger les champs ci-dessous avant export.")
        edited = st.data_editor(df, use_container_width=True, height=420, num_rows="dynamic")

        colA, colB, colC = st.columns(3)
        with colA:
            csv_bytes = edited.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Télécharger CSV", data=csv_bytes, file_name="menuiseries_extraction.csv", mime="text/csv")
        with colB:
            json_bytes = edited.to_json(orient="records", force_ascii=False).encode("utf-8")
            st.download_button("💾 Télécharger JSON", data=json_bytes, file_name="menuiseries_extraction.json", mime="application/json")
        with colC:
            st.caption("Conseil: gardez les dimensions en mm pour éviter les confusions.")

        if show_text and debug_texts:
            st.subheader("Texte brut (debug)")
            for fname, snippet in debug_texts:
                st.markdown(f"**{fname}**")
                st.text_area(f"Extrait: {fname}", value=snippet, height=200, key=f"txt_{fname}")
else:
    st.info("Ajoutez des PDF (ou un ZIP) pour lancer l’extraction.")
