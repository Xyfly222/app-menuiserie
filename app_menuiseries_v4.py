# app_menuiseries_v4.py
# Streamlit app: Extraction menuiseries (repère, quantité, largeur, hauteur, finition)
# v4: Quantité détectée depuis la colonne "Qté" (bande X autour de l'en-tête) + fallback regex
#
# Usage: streamlit run app_menuiseries_v4.py

import io
import os
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pdfplumber
import pandas as pd

st.set_page_config(page_title="Extraction menuiseries v4", page_icon="🪟", layout="wide")

# =====================
#      REGEX & UTILS
# =====================

X_CHARS = r"[xX×\*]"
NUM = r"\d{2,4}(?:[ \u00A0]\d{3})?(?:[.,]\d+)?"

DIMENSION_RE = re.compile(
    rf'(?P<w>{NUM})\s*{X_CHARS}\s*(?P<h>{NUM})(?:\s*(?P<unit>mm|cm|m))?\b'
)
DIM_TAGGED_RE = re.compile(
    rf'(?:L(?:arg)?\.?\s*[:=]?\s*(?P<w>{NUM})\s*(?P<unitw>mm|cm|m)?)'
    rf'.{{0,40}}?'
    rf'(?:H(?:aut)?\.?\s*[:=]?\s*(?P<h>{NUM})\s*(?P<unith>mm|cm|m)?)',
    re.I
)

# Quantity: regex fallback (when header band not detected)
QTE_PATTERNS = [
    re.compile(r'(?:Qt(?:e|é)|Quantit(?:e|é)|Qty)\s*[:\-]?\s*(?P<q>\d+(?:[.,]\d+)?)', re.I),
    re.compile(r'\b(?P<q>\d+(?:[.,]\d+)?)\s*(?:u|unit(?:e|é)s?|pcs?|pi[eè]ces?)\b', re.I),
    re.compile(r'(^|\b)(?P<q>\d{1,3}(?:[.,]\d+)?)\s*[xX]\b'),   # "2 x Fenêtre …"
    re.compile(r'\b[xX]\s*(?P<q>\d{1,3}(?:[.,]\d+)?)\b'),        # "Fenêtre x 2"
]

REPERE_PATTERNS = [
    re.compile(r'\bRep(?:[èe]re|\.?)\s*[:\-]?\s*(?P<r>[A-Za-z0-9][A-Za-z0-9\-_\/]{0,20})', re.I),
    re.compile(r'\bR(?:ep)?\s*[:\-]?\s*(?P<r>[A-Za-z0-9][A-Za-z0-9\-_\/]{0,20})\b', re.I),
    re.compile(r'\b(?P<r>[A-Z]{1,4}\d{1,4}(?:-[A-Z0-9]{1,4}){0,3})\b'),
]

FINITIONS = [
    r'RAL\s*\d{3,4}(?:\s*(?:MAT|MATTE|SATIN[ÉE]?|BRILLANT|TEXTUR[ÉE]))?',
    r'BLANC(?:\s+RAL\s*\d{3,4})?',
    r'GRIS(?:\s+ANTHRACITE)?',
    r'NOIR',
    r'ANODIS[ÉE]?',
    r'LAQU[ÉE]?',
    r'NATUREL',
    r'CH[ÊE]NE(?:\s+DOR[ÉE])?',
    r'ACAJOU',
    r'IVOIRE(?:\s*1015)?',
    r'7016', r'9005', r'9016', r'7022', r'1015'
]
FINIT_RE = re.compile("|".join(FINITIONS), re.I)

DIM_CLEAN = lambda s: re.sub(r"\s+(?=\d{3}\b)", "", s)

def _to_float(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    s = DIM_CLEAN(s)
    s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def _to_mm(v: Optional[float], unit: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    if not unit:
        return v  # assume mm
    u = unit.lower()
    if u == "mm":
        return v
    if u == "cm":
        return v * 10.0
    if u == "m":
        return v * 1000.0
    return v

def _cluster_rows(words: List[Dict[str, Any]], y_tol: float = 3.5) -> List[List[Dict[str, Any]]]:
    rows: List[List[Dict[str, Any]]] = []
    for w in sorted(words, key=lambda d: (round(d.get("top", 0.0), 1), d.get("x0", 0.0))):
        placed = False
        for row in rows:
            avg_top = sum(t["top"] for t in row) / len(row)
            if abs(w["top"] - avg_top) <= y_tol:
                row.append(w); placed = True; break
        if not placed:
            rows.append([w])
    for row in rows:
        row.sort(key=lambda d: d["x0"])
    return rows

def _join_row_text(row: List[Dict[str, Any]]) -> str:
    t = " ".join(w["text"] for w in row if w.get("text"))
    return re.sub(r"\s{2,}", " ", t).strip()

def _find_first(pats: List[re.Pattern], text: str, group: str) -> Optional[str]:
    for pat in pats:
        m = pat.search(text)
        if m and group in pat.groupindex:
            return m.group(group)
    return None

# Detect header word "Qté" band
HEADER_QTE_RE = re.compile(r'^(?:qt[eé]|qte|qty)$', re.I)

def _find_qte_band(rows: List[List[Dict[str, Any]]], pad: float = 10.0) -> Optional[Tuple[float, float]]:
    """
    Find the X band (x0..x1) of the header cell 'Qté'. Returns (x0-pad, x1+pad).
    """
    for row in rows:
        for w in row:
            txt = (w.get("text") or "").strip()
            if HEADER_QTE_RE.match(txt):
                x0, x1 = w["x0"], w["x1"]
                return (x0 - pad, x1 + pad)
    return None

NUM_ONLY_RE = re.compile(r'^\d{1,4}(?:[.,]\d+)?$')

def _quantity_from_band(row: List[Dict[str, Any]], band: Tuple[float, float]) -> Optional[float]:
    bx0, bx1 = band
    candidates: List[float] = []
    for w in row:
        x0, x1 = w["x0"], w["x1"]
        if x1 < bx0 or x0 > bx1:
            continue
        txt = (w.get("text") or "").strip()
        if NUM_ONLY_RE.match(txt):
            val = _to_float(txt)
            if val is not None:
                candidates.append(val)
    if candidates:
        # Choose the leftmost (or first) numeric as quantity
        return float(int(candidates[0]))  # quantité entière
    return None

def parse_layout_with_qte_column(file_bytes: bytes, source_name: str, y_tol: float, qte_pad: float, invert_dims: bool=False) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words() or []
            if not words:
                continue
            rows = _cluster_rows(words, y_tol=y_tol)
            qte_band = _find_qte_band(rows, pad=qte_pad)

            for row in rows:
                text = _join_row_text(row)

                # Dimensions
                largeur_mm = hauteur_mm = None
                m = DIMENSION_RE.search(text)
                if m:
                    w = _to_float(m.group("w")); h = _to_float(m.group("h"))
                    unit = m.group("unit")
                    largeur_mm = _to_mm(w, unit); hauteur_mm = _to_mm(h, unit)
                else:
                    m2 = DIM_TAGGED_RE.search(text)
                    if m2:
                        w = _to_float(m2.group("w")); h = _to_float(m2.group("h"))
                        uw = m2.group("unitw"); uh = m2.group("unith")
                        largeur_mm = _to_mm(w, uw); hauteur_mm = _to_mm(h, uh)

                if largeur_mm is None or hauteur_mm is None:
                    continue

                if invert_dims:
                    largeur_mm, hauteur_mm = hauteur_mm, largeur_mm

                # Quantité from column band, else fallback regex
                quantite = None
                if qte_band is not None:
                    quantite = _quantity_from_band(row, qte_band)
                if quantite is None:
                    q = _find_first(QTE_PATTERNS, text, "q")
                    if q:
                        try: quantite = float(q.replace(",", "."))
                        except: pass

                # Repère / Finition
                repere = _find_first(REPERE_PATTERNS, text, "r")
                fin = None
                mf = FINIT_RE.search(text)
                if mf: fin = mf.group(0)

                # Confiance
                conf = 0.0
                if largeur_mm is not None and hauteur_mm is not None: conf += 0.5
                if quantite is not None: conf += 0.3
                if repere: conf += 0.15
                if fin: conf += 0.05

                rows_out.append({
                    "fichier": os.path.basename(source_name),
                    "ligne_source": text,
                    "repere": repere,
                    "quantite": int(quantite) if isinstance(quantite, (int, float)) else None,
                    "largeur_mm": largeur_mm,
                    "hauteur_mm": hauteur_mm,
                    "finition": fin,
                    "confiance": round(conf, 2),
                })
    return rows_out

# =====================
#        UI
# =====================

st.title("🪟 Extraction menuiseries v4 — Qté par colonne")
st.caption("Détecte la Qté directement sous l’en-tête 'Qté' (colonne n°4), avec fallback regex.")

with st.sidebar:
    st.header("Paramètres")
    allow_zip = st.toggle("Autoriser fichiers ZIP", value=True)
    y_tol = st.slider("Tolérance d’alignement (Y)", min_value=2.0, max_value=8.0, value=3.5, step=0.5)
    qte_pad = st.slider("Largeur de bande autour de 'Qté' (px)", min_value=5.0, max_value=50.0, value=12.0, step=1.0)
    profil = st.selectbox("Profil fournisseur", ["Générique", "Inverser L/H"], index=0)
    invert_dims = (profil == "Inverser L/H")
    st.markdown("---")
    st.caption("Astuce: si la Qté n’est pas lue, augmentez la bande autour de 'Qté' ou la tolérance Y.")

accept_types = ["pdf"]
if allow_zip:
    accept_types.append("zip")

uploads = st.file_uploader("Déposez des PDF, ou un .zip de PDF", type=accept_types, accept_multiple_files=True)

all_rows: List[Dict[str, Any]] = []

if uploads:
    for up in uploads:
        name = up.name
        raw = up.read()

        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for info in zf.infolist():
                        if info.filename.lower().endswith(".pdf"):
                            pdf_bytes = zf.read(info)
                            rows = parse_layout_with_qte_column(pdf_bytes, info.filename, y_tol=y_tol, qte_pad=qte_pad, invert_dims=invert_dims)
                            all_rows.extend(rows)
            except zipfile.BadZipFile:
                st.error(f"Archive ZIP invalide: {name}")
        elif name.lower().endswith(".pdf"):
            rows = parse_layout_with_qte_column(raw, name, y_tol=y_tol, qte_pad=qte_pad, invert_dims=invert_dims)
            all_rows.extend(rows)

if not uploads:
    st.info("Ajoutez des fichiers pour commencer.")
else:
    if not all_rows:
        st.warning("Aucune ligne détectée. Ajustez les paramètres ou vérifiez la présence de l’en-tête 'Qté'.")
    else:
        df = pd.DataFrame(all_rows)
        df = df.sort_values(["fichier", "confiance"], ascending=[True, False]).reset_index(drop=True)
        st.success(f"{len(df)} ligne(s) détectée(s).")
        edited = st.data_editor(df, use_container_width=True, height=440, num_rows="dynamic")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Télécharger CSV",
                data=edited.to_csv(index=False).encode("utf-8"),
                file_name="menuiseries_extraction_v4.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                "💾 Télécharger JSON",
                data=edited.to_json(orient="records", force_ascii=False).encode("utf-8"),
                file_name="menuiseries_extraction_v4.json",
                mime="application/json"
            )

        with st.expander("ℹ️ Détails détection Qté"):
            st.markdown("""
- L’app cherche le mot **'Qté'** (ou **Qte/QTY**) dans l’en-tête et définit une **bande verticale** autour de cette colonne.
- La quantité est le **premier nombre seul** trouvé **dans cette bande** pour chaque rangée.
- Si l’en-tête n’est pas trouvé, on bascule sur un **fallback regex** (ex: `2 x`, `Qté: 2`, `2 u`).
- Vous pouvez élargir la bande avec le réglage **'Largeur de bande'** si nécessaire.
            """)
