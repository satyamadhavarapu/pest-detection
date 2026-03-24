"""
CropGuard - Crop Pest Detection App
Run: streamlit run pest_detector_app.py
"""

import os
import io
import warnings
import numpy as np
from datetime import datetime
from PIL import Image
import streamlit as st

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 CropGuard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
*, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero-title {
    font-size:2.6rem;font-weight:700;text-align:center;
    background:linear-gradient(90deg,#56ab2f,#a8e063);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.1rem;
}
.hero-sub{text-align:center;color:#888;font-size:0.95rem;margin-bottom:1.5rem;}
.card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.09);
    border-radius:14px;padding:1.3rem;margin-bottom:1rem;}
.card-title{font-size:1rem;font-weight:600;color:#a8e063;margin-bottom:0.7rem;}
.result-good{background:linear-gradient(135deg,#1a472a,#2d6a4f);border:2px solid #52b788;
    border-radius:14px;padding:1.4rem;text-align:center;margin-bottom:1rem;}
.result-bad{background:linear-gradient(135deg,#6b1a1a,#9e2a2b);border:2px solid #e63946;
    border-radius:14px;padding:1.4rem;text-align:center;margin-bottom:1rem;}
.result-unknown{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.15);
    border-radius:14px;padding:1.4rem;text-align:center;margin-bottom:1rem;}
.result-label{font-size:1.8rem;font-weight:700;color:#fff;margin:0.3rem 0;}
.result-sub{font-size:1rem;color:rgba(255,255,255,0.75);}
.cbar-green{height:9px;border-radius:5px;background:linear-gradient(90deg,#56ab2f,#a8e063);margin-top:0.5rem;}
.cbar-red{height:9px;border-radius:5px;background:linear-gradient(90deg,#e63946,#ff6b6b);margin-top:0.5rem;}
.badge{display:inline-block;padding:0.15rem 0.7rem;border-radius:20px;font-size:0.78rem;
    font-weight:600;background:rgba(86,171,47,0.15);border:1px solid #56ab2f;color:#a8e063;margin:0.15rem;}
.pre-item{background:rgba(255,255,255,0.04);border-left:3px solid #e63946;border-radius:7px;
    padding:0.6rem 0.9rem;margin:0.35rem 0;color:#eee;font-size:0.92rem;}
.prev-item{background:rgba(255,255,255,0.04);border-left:3px solid #56ab2f;border-radius:7px;
    padding:0.6rem 0.9rem;margin:0.35rem 0;color:#eee;font-size:0.92rem;}
.info-pill{background:rgba(86,171,47,0.1);border:1px solid rgba(86,171,47,0.3);
    border-radius:8px;padding:0.7rem 1rem;color:#a8e063;font-size:0.88rem;margin-top:0.5rem;}
.warn-pill{background:rgba(230,57,70,0.1);border:1px solid rgba(230,57,70,0.3);
    border-radius:8px;padding:0.7rem 1rem;color:#e63946;font-size:0.88rem;margin-top:0.5rem;}
.stButton>button{background:linear-gradient(90deg,#56ab2f,#a8e063);color:#0f2027;
    font-weight:700;border:none;border-radius:10px;padding:0.55rem 1.5rem;font-size:0.97rem;width:100%;}
.stButton>button:hover{box-shadow:0 6px 18px rgba(86,171,47,0.4);}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CLASS NAME MAPS
# ══════════════════════════════════════════════════════════════════════════════

# Your Keras models (CNN, MobileNetV2, EfficientNetB0) have 12 output units.
# These are the class names IN ORDER as your model was trained.
# ⚠️ If predictions are wrong, reorder this list to match your training data.yaml / folder order.
KERAS_CLASS_NAMES = [
    "ants",          # 0
    "bees",          # 1
    "beetles",       # 2
    "caterpillars",  # 3
    "earthworms",    # 4
    "earwigs",       # 5
    "grasshoppers",  # 6
    "moths",         # 7
    "slugs",         # 8
    "snails",        # 9
    "wasps",         # 10
    "weevils",       # 11
]

# Your YOLOv8 model's actual class names (from debug: {0:'Ants', 1:'Bees', ...})
# Mapped to our PEST_DB keys
YOLO_TO_DB = {
    "ants":        "unknown",       # no ants in PEST_DB — treat as unknown
    "bees":        "honeybee",
    "beetles":     "ladybug",       # closest match
    "caterpillars":"armyworm",
    "earthworms":  "unknown",
    "earwigs":     "unknown",
    "grasshoppers":"grasshopper",
    "moths":       "armyworm",      # moths → armyworm (similar damage)
    "slugs":       "unknown",
    "snails":      "unknown",
    "wasps":       "unknown",
    "weevils":     "stem_borer",    # weevils bore into stems
}

# ══════════════════════════════════════════════════════════════════════════════
# PEST DATABASE
# ══════════════════════════════════════════════════════════════════════════════
PEST_DB = {
    "ants": {
        "type": "BAD",
        "common_name": "Ants",
        "emoji": "🐜",
        "description": "Ants can damage roots and protect aphids that harm crops.",
        "precautions": ["Inspect soil around plant roots.",
                        "Avoid excess organic debris near crops."],
        "preventions": ["Use boric acid bait traps.",
                        "Apply ant-specific insecticides if infestation is severe."],
        "pesticide": "Boric Acid, Chlorpyrifos",
        "severity": "Medium",
    },
    "bees": {
        "type": "GOOD",
        "common_name": "Bees",
        "emoji": "🐝",
        "description": "Bees are essential pollinators that increase crop yield.",
        "benefit": "Improves pollination and increases production by 20-30%.",
        "how_to_encourage": ["Plant flowering crops.",
                             "Avoid spraying pesticides during flowering."],
    },
    "beetles": {
        "type": "BAD",
        "common_name": "Beetles",
        "emoji": "🐞",
        "description": "Some beetles feed on leaves, stems and fruits.",
        "precautions": ["Monitor leaves for chewing damage."],
        "preventions": ["Use neem oil spray.",
                        "Introduce natural predators."],
        "pesticide": "Imidacloprid, Carbaryl",
        "severity": "Medium",
    },
    "caterpillars": {
        "type": "BAD",
        "common_name": "Caterpillars",
        "emoji": "🪱",
        "description": "Caterpillars consume leaves rapidly causing severe crop damage.",
        "precautions": ["Inspect underside of leaves for eggs."],
        "preventions": ["Apply Bacillus thuringiensis (Bt).",
                        "Use Spinosad spray."],
        "pesticide": "Bt, Spinosad",
        "severity": "High",
    },
    "earthworms": {
        "type": "GOOD",
        "common_name": "Earthworms",
        "emoji": "🪱",
        "description": "Earthworms improve soil fertility and aeration.",
        "benefit": "Enhances soil structure and nutrient cycling.",
        "how_to_encourage": ["Add organic compost.",
                             "Avoid heavy chemical use."],
    },
    "earwigs": {
        "type": "BAD",
        "common_name": "Earwigs",
        "emoji": "🪳",
        "description": "Earwigs feed on leaves and flowers.",
        "precautions": ["Check moist areas of field."],
        "preventions": ["Use traps with vegetable oil.",
                        "Apply insecticide if severe."],
        "pesticide": "Carbaryl",
        "severity": "Medium",
    },
    "grasshoppers": {
        "type": "BAD",
        "common_name": "Grasshoppers",
        "emoji": "🦗",
        "description": "Grasshoppers chew leaves and stems aggressively.",
        "precautions": ["Monitor field edges."],
        "preventions": ["Apply Malathion spray.",
                        "Use biological control Nosema locustae."],
        "pesticide": "Malathion",
        "severity": "High",
    },
    "moths": {
        "type": "BAD",
        "common_name": "Moths",
        "emoji": "🦋",
        "description": "Moth larvae damage crops by feeding on leaves.",
        "precautions": ["Install pheromone traps."],
        "preventions": ["Use Bt spray."],
        "pesticide": "Bt, Spinosad",
        "severity": "Medium",
    },
    "slugs": {
        "type": "BAD",
        "common_name": "Slugs",
        "emoji": "🐌",
        "description": "Slugs feed on leaves and young plants.",
        "precautions": ["Remove plant debris."],
        "preventions": ["Use slug bait pellets."],
        "pesticide": "Metaldehyde",
        "severity": "Medium",
    },
    "snails": {
        "type": "BAD",
        "common_name": "Snails",
        "emoji": "🐌",
        "description": "Snails damage seedlings and leaves.",
        "precautions": ["Check during night time."],
        "preventions": ["Use copper barriers."],
        "pesticide": "Metaldehyde",
        "severity": "Medium",
    },
    "wasps": {
        "type": "GOOD",
        "common_name": "Wasps",
        "emoji": "🐝",
        "description": "Many wasps act as biological control agents.",
        "benefit": "Control caterpillars and aphids naturally.",
        "how_to_encourage": ["Maintain biodiversity in field."],
    },
    "weevils": {
        "type": "BAD",
        "common_name": "Weevils",
        "emoji": "🐛",
        "description": "Weevils bore into stems and grains causing severe loss.",
        "precautions": ["Inspect stem bases."],
        "preventions": ["Apply appropriate insecticide.",
                        "Practice crop rotation."],
        "pesticide": "Chlorantraniliprole",
        "severity": "High",
    },
    "unknown": {
    "type": "UNKNOWN",
    "common_name": "Unknown Pest",
    "emoji": "❓",
    "description": "The model could not confidently identify this insect.",
    "precautions": ["Please upload a clearer image."],
    "preventions": ["Consult agricultural expert."],
    "pesticide": "N/A",
    "severity": "Unknown",
},
}

def lookup_pest(label: str) -> dict:
    n = str(label).lower().strip().replace(" ","_").replace("-","_")
    return PEST_DB.get(n, PEST_DB["unknown"])

# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGES
# ══════════════════════════════════════════════════════════════════════════════
LANGUAGES = {
    "English":"en","Hindi (हिन्दी)":"hi","Bengali (বাংলা)":"bn",
    "Telugu (తెలుగు)":"te","Marathi (मराठी)":"mr","Tamil (தமிழ்)":"ta",
    "Urdu (اردو)":"ur","Gujarati (ગુજરાતી)":"gu","Kannada (ಕನ್ನಡ)":"kn",
    "Odia (ଓଡ଼ିଆ)":"or","Malayalam (മലയാളം)":"ml","Punjabi (ਪੰਜਾਬੀ)":"pa",
    "Assamese (অসমীয়া)":"as","Maithili (मैथिली)":"mai","Nepali (नेपाली)":"ne",
    "Sindhi (سنڌي)":"sd","Konkani (कोंकणी)":"gom","Kashmiri (کٲشُر)":"ks",
    "Dogri (डोगरी)":"doi","Manipuri (মৈতৈলোন্)":"mni","Sanskrit (संस्कृतम्)":"sa",
}

@st.cache_resource(show_spinner=False)
def _get_translator():
    try:
        from googletrans import Translator
        return Translator()
    except Exception:
        return None

def tx(text: str, lang: str) -> str:
    if lang == "en" or not text:
        return text
    t = _get_translator()
    if not t:
        return text
    try:
        return t.translate(str(text), dest=lang).text
    except Exception:
        return text

def tx_list(items, lang):
    return [tx(i, lang) for i in items]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  — fixed for Keras 3 / older TF compatibility
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _load_keras(path: str):
    """Load .keras model saved with Keras 3 into older TF/Keras installs."""
    import logging
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    # Try modern load first
    for kwargs in [
        {"compile": False},
        {"compile": False, "safe_mode": False},
        {},
    ]:
        try:
            return tf.keras.models.load_model(path, **kwargs)
        except Exception:
            continue

    # Last resort: load weights only via h5 export if possible
    return None

@st.cache_resource(show_spinner=False)
def _load_yolo(path: str):
    import logging
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    try:
        from ultralytics import YOLO
        return YOLO(path)
    except Exception as e:
        st.sidebar.error(f"YOLO load failed: {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def _infer_keras(path, img, class_names):
    if not os.path.exists(path):
        return None
    m = _load_keras(path)
    if m is None:
        return None
    is_eff = "efficientnet" in os.path.basename(path).lower()
    arr = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
    if not is_eff:
        arr /= 255.0
    preds = m.predict(np.expand_dims(arr, 0), verbose=0)[0]
    idx   = int(np.argmax(preds))
    label = class_names[idx] if idx < len(class_names) else f"class_{idx}"
    return {"label": label, "confidence": float(preds[idx]),
            "all_probs": preds.tolist(), "raw_index": idx}

def _infer_yolo(path, img):
    if not os.path.exists(path):
        return None
    m = _load_yolo(path)
    if m is None:
        return None
    results = m(np.array(img.convert("RGB")), verbose=False)
    r = results[0]

    if r.probs is not None:                          # classification
        probs = r.probs.data.cpu().numpy()
        idx   = int(np.argmax(probs))
        raw_name = r.names.get(idx, f"class_{idx}").lower()
        db_label = raw_name
        return {"label": db_label, "confidence": float(probs[idx]),
                "all_probs": probs.tolist(), "raw_index": idx,
                "yolo_raw": raw_name}

    if len(r.boxes):                                 # detection
        box      = r.boxes[0]
        idx      = int(box.cls.item())
        raw_name = r.names.get(idx, f"class_{idx}").lower()
        db_label = YOLO_TO_DB.get(raw_name, "unknown")
        return {"label": db_label, "confidence": float(box.conf.item()),
                "all_probs": [], "raw_index": idx, "yolo_raw": raw_name}
    return None

def run_single(model_dir, choice, img):
    fmap = {
        "CNN":            "cnn_best.keras",
        "MobileNetV2":    "mobilenetv2_best.keras",
        "EfficientNetB0": "efficientnetb0_best.keras",
    }
    if choice in fmap:
        res = _infer_keras(os.path.join(model_dir, fmap[choice]), img, KERAS_CLASS_NAMES)
    else:
        res = _infer_yolo(os.path.join(model_dir, "best.pt"), img)

    if res is None:
        return {"label":"unknown","confidence":0.0,"all_probs":[],"raw_index":-1,"models_used":[choice]}
    res["models_used"] = [choice]
    return res

def run_ensemble(model_dir, img):
    results, used = [], []

    for name, fname in [("CNN","cnn_best.keras"),
                         ("MobileNetV2","mobilenetv2_best.keras"),
                         ("EfficientNetB0","efficientnetb0_best.keras")]:
        res = _infer_keras(os.path.join(model_dir, fname), img, KERAS_CLASS_NAMES)
        if res:
            results.append(res); used.append(name)

    yolo_res = _infer_yolo(os.path.join(model_dir, "best.pt"), img)
    if yolo_res:
        results.append(yolo_res); used.append("YOLOv8")

    if not results:
        return {"label":"unknown","confidence":0.0,"all_probs":[],"raw_index":-1,"models_used":[]}

    # Vote by label (majority wins)
    from collections import Counter
    label_votes = Counter(r["label"] for r in results)
    winner = label_votes.most_common(1)[0][0]
    # Average confidence of models that voted for winner
    winner_confs = [r["confidence"] for r in results if r["label"] == winner]
    avg_conf = float(np.mean(winner_confs))

    # Best all_probs from keras models only (same class space)
    keras_probs = [r["all_probs"] for r in results
                   if r["all_probs"] and len(r["all_probs"]) == len(KERAS_CLASS_NAMES)]
    all_probs = np.mean(keras_probs, axis=0).tolist() if keras_probs else []

    return {"label": winner, "confidence": avg_conf,
            "all_probs": all_probs, "raw_index": -1, "models_used": used}

# ══════════════════════════════════════════════════════════════════════════════
# PDF
# ══════════════════════════════════════════════════════════════════════════════
def make_pdf(img, pred, pest_info, lang_name, lang_code, timestamp):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Image as RLImage, Table, TableStyle, HRFlowable)
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm,rightMargin=2*cm,
                             topMargin=2*cm,bottomMargin=2*cm)

    def S(n,**kw): return ParagraphStyle(n,**kw)
    Tc=S("T",fontSize=20,fontName="Helvetica-Bold",alignment=TA_CENTER,
          textColor=colors.HexColor("#1b5e20"),spaceAfter=6)
    Sc=S("S",fontSize=11,alignment=TA_CENTER,textColor=colors.HexColor("#555"),spaceAfter=4)
    Hc=S("H",fontSize=13,fontName="Helvetica-Bold",textColor=colors.HexColor("#2e7d32"),
          spaceBefore=12,spaceAfter=4)
    Bc=S("B",fontSize=10,leading=14,textColor=colors.HexColor("#333"))
    Wc=S("W",fontSize=10,leading=14,textColor=colors.HexColor("#b71c1c"))
    Gc=S("G",fontSize=10,leading=14,textColor=colors.HexColor("#1b5e20"))
    Fc=S("F",fontSize=8,textColor=colors.grey,alignment=TA_CENTER)

    story=[]
    story.append(Paragraph("CropGuard Pest Detection Report",Tc))
    story.append(Paragraph(f"Generated: {timestamp}  |  Language: {lang_name}",Sc))
    story.append(HRFlowable(width="100%",thickness=2,color=colors.HexColor("#2e7d32")))
    story.append(Spacer(1,12))

    ibuf=io.BytesIO(); t=img.convert("RGB"); t.thumbnail((400,400)); t.save(ibuf,"JPEG"); ibuf.seek(0)
    story.append(RLImage(ibuf,width=10*cm,height=10*cm))
    story.append(Spacer(1,10))

    pt=pest_info.get("type","UNKNOWN")
    pnam=tx(pest_info.get("common_name",pred["label"]),lang_code)
    stat=tx("BENEFICIAL" if pt=="GOOD" else "HARMFUL" if pt=="BAD" else "UNKNOWN",lang_code)
    bg=(colors.HexColor("#e8f5e9") if pt=="GOOD" else
        colors.HexColor("#ffebee") if pt=="BAD" else colors.HexColor("#f5f5f5"))

    tbl=Table([
        [tx("Pest",lang_code),tx("Status",lang_code),tx("Confidence",lang_code),tx("Model(s)",lang_code)],
        [pnam,stat,f"{pred['confidence']*100:.1f}%",", ".join(pred.get("models_used",["N/A"]))],
    ],colWidths=[4*cm,4*cm,3*cm,6*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#2e7d32")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("BACKGROUND",(0,1),(-1,1),bg),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("GRID",(0,0),(-1,-1),0.5,colors.grey),
        ("TOPPADDING",(0,0),(-1,-1),8),("BOTTOMPADDING",(0,0),(-1,-1),8),
    ]))
    story.append(tbl); story.append(Spacer(1,10))
    story.append(Paragraph(tx("Description",lang_code),Hc))
    story.append(Paragraph(tx(pest_info.get("description",""),lang_code),Bc))

    if pt=="BAD":
        story.append(Spacer(1,6))
        story.append(Paragraph(
            f"{tx('Severity',lang_code)}: <b>{tx(pest_info.get('severity',''),lang_code)}</b>"
            f"  |  {tx('Pesticide',lang_code)}: <b>{pest_info.get('pesticide','')}</b>",Bc))
        story.append(Paragraph(tx("Precautions",lang_code),Hc))
        for i in tx_list(pest_info.get("precautions",[]),lang_code):
            story.append(Paragraph(f"• {i}",Wc))
        story.append(Paragraph(tx("Prevention & Treatment",lang_code),Hc))
        for i in tx_list(pest_info.get("preventions",[]),lang_code):
            story.append(Paragraph(f"• {i}",Gc))
    elif pt=="GOOD":
        story.append(Paragraph(tx("Benefit to Farm",lang_code),Hc))
        story.append(Paragraph(tx(pest_info.get("benefit",""),lang_code),Gc))
        story.append(Paragraph(tx("How to Encourage",lang_code),Hc))
        for i in tx_list(pest_info.get("how_to_encourage",[]),lang_code):
            story.append(Paragraph(f"• {i}",Gc))

    story.append(Spacer(1,20))
    story.append(HRFlowable(width="100%",thickness=1,color=colors.grey))
    story.append(Paragraph(tx(
        "CropGuard AI Report | Consult your local Agricultural Extension Officer for expert advice.",
        lang_code),Fc))
    doc.build(story)
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown('<div class="hero-title">🌾 CropGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">AI-Powered Crop Pest Detection & Advisory System</div>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        st.markdown("---")
        st.markdown("### 🤖 Model")
        MODEL_CHOICES = ["CNN","MobileNetV2","EfficientNetB0","YOLOv8","🔥 Ensemble (All)"]
        model_choice  = st.radio("Choose:", MODEL_CHOICES, index=4)
        st.markdown("---")
        st.markdown("### 🌐 Language")
        lang_name = st.selectbox("Translate to:", list(LANGUAGES.keys()), index=0)
        lang_code = LANGUAGES[lang_name]
        st.markdown("---")
        st.markdown("""<div class="info-pill">
        <b>pip install:</b><br>
        streamlit tensorflow ultralytics<br>
        pillow googletrans==4.0.0rc1 reportlab
        </div>""", unsafe_allow_html=True)

    col_in, col_out = st.columns([1,1], gap="large")

    with col_in:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📸 Input Image</div>', unsafe_allow_html=True)
        mode = st.radio("", ["📤 Upload Image","📷 Live Camera"],
                        horizontal=True, label_visibility="collapsed")
        img = None
        if mode == "📤 Upload Image":
            up = st.file_uploader("", type=["jpg","jpeg","png","webp","bmp"],
                                   label_visibility="collapsed")
            if up:
                img = Image.open(up)
                st.image(img, use_column_width=True, caption="Uploaded image")
        else:
            cam = st.camera_input("", label_visibility="collapsed")
            if cam:
                img = Image.open(cam)
                st.image(img, use_column_width=True, caption="Captured image")
        if img:
            st.caption(f"{img.size[0]}×{img.size[1]}px")
        st.markdown('</div>', unsafe_allow_html=True)
        clicked = st.button("🔍 Analyze Pest", disabled=(img is None))

    with col_out:
        if img is not None and clicked:
            with st.spinner("🔬 Analysing image…"):
                ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                pred = (run_ensemble(model_dir, img)
                        if model_choice == "🔥 Ensemble (All)"
                        else run_single(model_dir, model_choice, img))

            raw   = pred.get("label","unknown")
            info  = lookup_pest(raw)
            ptype = info.get("type","UNKNOWN")
            tname = tx(info.get("common_name", raw), lang_code)
            tdesc = tx(info.get("description",""), lang_code)
            conf  = pred.get("confidence",0.0) * 100

            if ptype == "GOOD":
                st.markdown(f"""
                <div class="result-good">
                  <div style="font-size:2.5rem">{info['emoji']}</div>
                  <div class="result-label">{tname}</div>
                  <div class="result-sub">✅ {tx("Beneficial Pest",lang_code)}</div>
                  <div style="color:rgba(255,255,255,0.6);font-size:0.85rem;margin-top:0.4rem">
                    {tx("Confidence",lang_code)}: {conf:.1f}%</div>
                  <div class="cbar-green" style="width:{conf:.0f}%;margin:0.4rem auto"></div>
                </div>""", unsafe_allow_html=True)

            elif ptype == "BAD":
                sev = tx(info.get("severity",""), lang_code)
                st.markdown(f"""
                <div class="result-bad">
                  <div style="font-size:2.5rem">{info['emoji']}</div>
                  <div class="result-label">{tname}</div>
                  <div class="result-sub">⚠️ {tx("Harmful Pest",lang_code)}</div>
                  <div style="color:rgba(255,255,255,0.6);font-size:0.85rem;margin-top:0.4rem">
                    {tx("Confidence",lang_code)}: {conf:.1f}%</div>
                  <div class="cbar-red" style="width:{conf:.0f}%;margin:0.4rem auto"></div>
                  <div style="margin-top:0.5rem;display:inline-block;padding:0.2rem 0.8rem;
                              background:rgba(0,0,0,0.25);border-radius:20px;
                              color:#ffcdd2;font-size:0.82rem">
                    {tx("Severity",lang_code)}: {sev}</div>
                </div>""", unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="result-unknown">
                  <div style="font-size:2.5rem">❓</div>
                  <div class="result-label" style="color:#aaa">{tname}</div>
                  <div style="color:#777;font-size:0.85rem;margin-top:0.4rem">
                    Raw: <code>{raw}</code> — check KERAS_CLASS_NAMES order in the code.</div>
                </div>""", unsafe_allow_html=True)

            badges = "".join([f'<span class="badge">{m}</span>'
                               for m in pred.get("models_used",[model_choice])])
            st.markdown(badges, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="card" style="margin-top:0.8rem">
              <div class="card-title">📋 {tx("Description",lang_code)}</div>
              <div style="color:#ccc;font-size:0.93rem">{tdesc}</div>
            </div>""", unsafe_allow_html=True)

        elif img is None:
            st.markdown("""
            <div class="card" style="text-align:center;padding:3rem;color:#555">
              <div style="font-size:3.5rem">🌿</div>
              <div style="margin-top:0.8rem">Upload or capture an image to begin</div>
            </div>""", unsafe_allow_html=True)

    # ── Full-width details ────────────────────────────────────────────────────
    if img is not None and clicked and "pred" in dir():
        raw   = pred.get("label","unknown")
        info  = lookup_pest(raw)
        ptype = info.get("type","UNKNOWN")
        st.markdown("---")

        if ptype == "BAD":
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">⚠️ {tx("Precautions",lang_code)}</div>',
                            unsafe_allow_html=True)
                for item in tx_list(info.get("precautions",[]), lang_code):
                    st.markdown(f'<div class="pre-item">⚡ {item}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="warn-pill">💊 <b>{tx("Pesticide",lang_code)}:</b> {info.get("pesticide","")}</div>',
                    unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">✅ {tx("Prevention & Treatment",lang_code)}</div>',
                            unsafe_allow_html=True)
                for item in tx_list(info.get("preventions",[]), lang_code):
                    st.markdown(f'<div class="prev-item">🌱 {item}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        elif ptype == "GOOD":
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">🌟 {tx("Benefit",lang_code)}</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="prev-item">{tx(info.get("benefit",""),lang_code)}</div>',
                            unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">🌱 {tx("How to Encourage",lang_code)}</div>',
                            unsafe_allow_html=True)
                for item in tx_list(info.get("how_to_encourage",[]), lang_code):
                    st.markdown(f'<div class="prev-item">• {item}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Confidence chart (Keras only — consistent class space)
        all_probs = pred.get("all_probs",[])
        if len(all_probs) == len(KERAS_CLASS_NAMES):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="card-title">📊 {tx("Confidence Distribution",lang_code)}</div>',
                        unsafe_allow_html=True)
            st.bar_chart({KERAS_CLASS_NAMES[i]: float(all_probs[i])
                          for i in range(len(KERAS_CLASS_NAMES))})
            st.markdown('</div>', unsafe_allow_html=True)

        # PDF
        st.markdown(f'<div class="card-title">📄 {tx("Download Report",lang_code)}</div>',
                    unsafe_allow_html=True)
        try:
            with st.spinner("Generating PDF…"):
                pdf = make_pdf(img, pred, info, lang_name, lang_code, ts)
            st.download_button(
                label=f"⬇️ {tx('Download PDF Report',lang_code)}",
                data=pdf,
                file_name=f"CropGuard_{raw}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.markdown(f'<div class="info-pill">✅ {ts} | {lang_name}</div>',
                        unsafe_allow_html=True)
        except ImportError:
            st.markdown('<div class="warn-pill">pip install reportlab</div>',
                        unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="warn-pill">PDF error: {e}</div>',
                        unsafe_allow_html=True)

    st.markdown("""<hr><div style="text-align:center;color:#444;font-size:0.8rem;padding:0.5rem">
        CropGuard AI • TensorFlow + YOLOv8 + Streamlit
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
