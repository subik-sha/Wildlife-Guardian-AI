"""Wildlife Guardian AI - upgraded prototype (Step A + B + C).

Keeps the original SQLite tables (detection_history, alert_history) so
existing records still load. Distance is SIMULATED; no GPS/IoT is used.
"""
import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

# set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="Wildlife Guardian AI",
    page_icon="🐘",
    layout="wide",
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wildlife_history.db")
TIME_FMT = "%d-%m-%Y %I:%M:%S %p"
ALERT_KM = 2.0
MAX_UPLOAD_MB = 10
WILDLIFE_WORDS = (
    "lion", "tiger", "leopard", "jaguar", "cheetah", "elephant", "tusker", "zebra",
    "bear", "hippopotamus", "impala", "gazelle", "hartebeest", "ibex", "bighorn",
    "warthog", "boar", "buffalo", "giraffe", "deer", "wolf", "fox", "hyena", "monkey",
    "gorilla", "chimpanzee", "baboon", "rhinoceros", "antelope", "coyote", "dingo",
)

# ---------- Styling (professional forest theme; works offline) ----------
st.markdown(
    """
    <style>
    :root {
        --forest: #153B2E;
        --forest-deep: #0D241C;
        --leaf: #2E6B4F;
        --mint: #DCEFE3;
        --cream: #F7F5ED;
        --gold: #D5A84F;
        --ink: #20352B;
        --muted: #708278;
    }

    /* Main canvas and typography */
    .stApp {
        background:
            radial-gradient(ellipse at top right, rgba(213,168,79,.09), transparent 34%),
            linear-gradient(180deg, #F8FAF7 0%, #F2F6F1 100%);
        color: var(--ink);
    }
    .block-container {
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        font-family: Georgia, 'Times New Roman', serif;
        letter-spacing: -.35px;
        color: var(--forest-deep);
    }
    h1 { font-size: clamp(2rem, 3vw, 2.8rem); }
    h2 { margin-top: .5rem; }
    p, label, [data-testid="stCaptionContainer"] {
        line-height: 1.55;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #102D22 0%, #183E30 100%);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    section[data-testid="stSidebar"] * {
        color: #F4F7F2;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #B9CDBF;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,.18);
    }

    /* Cards and metric tiles */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E0E9E1;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 5px 18px rgba(21,59,46,.055);
    }
    [data-testid="stMetricLabel"] {
        color: #65796D;
        font-size: .9rem;
    }
    [data-testid="stMetricValue"] {
        color: var(--forest);
        font-weight: 750;
    }
    .kpi {
        background: linear-gradient(145deg, #153B2E, #214F3C);
        border: 1px solid #2F654D;
        border-radius: 16px;
        padding: 18px 20px;
        min-height: 108px;
        box-shadow: 0 8px 20px rgba(21,59,46,.10);
    }
    .kpi .label {
        color: #C4D9CB;
        font-size: .85rem;
        margin-bottom: 8px;
    }
    .kpi .value {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: clamp(1.65rem, 2.5vw, 2.2rem);
        line-height: 1.2;
        font-weight: 700;
        color: #FFFDF5;
    }
    .kpi.warn {
        border-left: 4px solid var(--gold);
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid #24563F;
        padding: .62rem 1.1rem;
        font-weight: 650;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1D5A3D, #2F7952);
        color: #FFFFFF;
        border: 0;
        box-shadow: 0 5px 13px rgba(29,90,61,.18);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 7px 16px rgba(21,59,46,.12);
        border-color: var(--gold);
    }

    /* Inputs, uploads, tables and charts */
    [data-testid="stFileUploader"] section {
        background: #FFFFFF;
        border: 1px dashed #A8C4B1;
        border-radius: 14px;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #E0E9E1;
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stExpander"] {
        background: rgba(255,255,255,.72);
        border: 1px solid #E0E9E1;
        border-radius: 12px;
    }
    hr {
        border-color: #DFE8E0;
        margin: 1.5rem 0;
    }
    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .7px;
        border: 1px solid #D5A84F;
        color: #805B16;
        background: #FFF8E7;
    }
    .alert-box {
        border-radius: 14px;
        padding: 18px 20px;
        font-size: 1.05rem;
        font-weight: 600;
        box-shadow: 0 5px 16px rgba(21,59,46,.06);
    }
    .alert-on {
        background: #FFF0EB;
        border: 1px solid #D96D54;
        color: #8B2E20;
    }
    .alert-off {
        background: #EAF6EE;
        border: 1px solid #77B28A;
        color: #205B37;
    }

    /* Make images feel integrated into the interface */
    [data-testid="stImage"] img {
        border-radius: 14px;
    }

    @media (max-width: 768px) {
        .block-container { padding: 1.2rem 1rem 2rem; }
        .kpi { padding: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi(label, value, warn=False):
    cls = "kpi warn" if warn else "kpi"
    st.markdown(
        f'<div class="{cls}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


# ---------- Database (same schema as before) ----------
def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_time TEXT NOT NULL, animal_name TEXT NOT NULL,
                confidence REAL NOT NULL, status TEXT NOT NULL)"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_time TEXT NOT NULL, distance_km REAL NOT NULL,
                alert_status TEXT NOT NULL)"""
        )


init_db()


def load_detections():
    with db() as conn:
        rows = conn.execute(
            "SELECT date_time, animal_name, confidence, status "
            "FROM detection_history ORDER BY id DESC"
        ).fetchall()
    df = pd.DataFrame(rows, columns=["Date & Time", "Result", "Confidence (%)", "Status"])
    return df


def load_alerts():
    with db() as conn:
        rows = conn.execute(
            "SELECT date_time, distance_km, alert_status FROM alert_history ORDER BY id DESC"
        ).fetchall()
    return pd.DataFrame(rows, columns=["Date & Time", "Distance (km)", "Alert Status"])


def save_detection(result, confidence, status):
    with db() as conn:
        conn.execute(
            "INSERT INTO detection_history (date_time, animal_name, confidence, status) "
            "VALUES (?, ?, ?, ?)",
            (datetime.now().strftime(TIME_FMT), result, confidence, status),
        )


def save_alert(distance_km):
    with db() as conn:
        conn.execute(
            "INSERT INTO alert_history (date_time, distance_km, alert_status) VALUES (?, ?, ?)",
            (datetime.now().strftime(TIME_FMT), distance_km, "Wildlife Alert (simulated)"),
        )


# ---------- Models ----------
@st.cache_resource(show_spinner="Loading detection model...")
def load_detector():
    from ultralytics import YOLOWorld

    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(["animal"])  # finds WHERE an animal is; the classifier names it
    return model


@st.cache_resource(show_spinner="Loading classifier...")
def load_classifier():
    from ultralytics import YOLO

    return YOLO("yolo11n-cls.pt")


def nice(name):
    return name.replace("_", " ").title()


def is_wildlife(name):
    n = name.lower().replace("_", " ")
    return any(w in n for w in WILDLIFE_WORDS)


def classify(img):
    """Return the top-5 (name, confidence 0-1) guesses of the classifier."""
    r = load_classifier()(img, verbose=False)[0]
    ids = [int(i) for i in r.probs.top5]
    confs = [float(c) for c in r.probs.top5conf]
    return [(r.names[i], c) for i, c in zip(ids, confs)]


def read_image(uploaded):
    """Validate and open an upload. Returns (image, error_message)."""
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"File is larger than {MAX_UPLOAD_MB} MB. Upload a smaller image."
    try:
        img = Image.open(uploaded)
        img.load()
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None, "This file could not be read as an image. Upload a JPG or PNG."
    img.thumbnail((1280, 1280))  # keeps detection fast on large photos
    return img, None


def draw_boxes(image, items):
    """items: list of ((x1, y1, x2, y2), label). Returns a new PIL image."""
    out = image.copy()
    d = ImageDraw.Draw(out)
    size = max(16, out.width // 40)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    for (x1, y1, x2, y2), label in items:
        d.rectangle([x1, y1, x2, y2], outline=(111, 165, 138), width=max(3, out.width // 250))
        tw = d.textlength(label, font=font)
        d.rectangle([x1, max(0, y1 - size - 6), x1 + tw + 10, max(0, y1 - size - 6) + size + 6],
                    fill=(111, 165, 138))
        d.text((x1 + 5, max(0, y1 - size - 6) + 2), label, fill=(16, 26, 22), font=font)
    return out


def run_detection(image, conf):
    """Step 1: YOLO-World finds animal boxes. Step 2: classifier names each box."""
    det = load_detector().predict(image, conf=conf, verbose=False)[0]
    boxes = sorted(det.boxes, key=lambda b: float(b.conf[0]), reverse=True)[:5]
    W, H = image.size
    rows, drawn = [], []

    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        pad = int(0.05 * max(x2 - x1, y2 - y1))
        crop = image.crop((max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)))
        name, c = classify(crop)[0]
        if is_wildlife(name) and c >= 0.25:
            label, species_conf = nice(name), round(c * 100, 1)
        else:
            label, species_conf = "Unidentified animal", 0.0
        rows.append({"Animal": label, "Species confidence (%)": species_conf,
                     "Detection confidence (%)": round(float(b.conf[0]) * 100, 1)})
        drawn.append(((x1, y1, x2, y2), label))

    whole = classify(image)
    note = ""
    if not rows:  # detector found no box: fall back to the whole-image classifier
        name, c = whole[0]
        if is_wildlife(name) and c >= 0.30:
            rows.append({"Animal": nice(name), "Species confidence (%)": round(c * 100, 1),
                         "Detection confidence (%)": None})
            note = "No box was found, so the whole image was classified instead."

    rows.sort(key=lambda r: (r["Species confidence (%)"], r["Detection confidence (%)"] or 0),
              reverse=True)
    debug = {"boxes_found": len(boxes), "whole_image_top3": [(nice(n), round(c * 100, 1)) for n, c in whole[:3]]}
    return rows, draw_boxes(image, drawn), note, debug


# ---------- Sidebar navigation ----------
st.sidebar.title("🐘 Wildlife Guardian")
st.sidebar.caption("AI wildlife detection and safety alerts")
page = st.sidebar.radio("Go to", ["Dashboard", "Detect", "Alerts", "History", "About"],
                        label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Prototype: works on uploaded images. Distance is simulated.")

if "last_detection" not in st.session_state:
    st.session_state.last_detection = None

# ---------- Pages ----------
if page == "Dashboard":
    st.title("Monitoring dashboard")
    det, alerts = load_detections(), load_alerts()
    total = len(det)
    with_animal = int((det["Status"] == "Animal detected").sum()) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Total scans", total)
    with c2: kpi("Scans with animals", with_animal)
    with c3: kpi("No animal found", total - with_animal)
    with c4: kpi("Alerts saved (simulated)", len(alerts), warn=True)

    st.write("")
    if total == 0:
        st.info("No scans yet. Open **Detect**, upload an image and run a scan.")
    else:
        left, right = st.columns(2)
        found = det[det["Status"] == "Animal detected"]
        with left:
            st.subheader("Detections by animal")
            if len(found):
                counts = found["Result"].str.replace(" animal", "", case=False, regex=False).str.title().str.split(", ").explode().value_counts()
                st.bar_chart(counts)
            else:
                st.caption("No animals detected yet.")
        with right:
            st.subheader("Scans per day")
            days = pd.to_datetime(det["Date & Time"], format=TIME_FMT, errors="coerce").dt.date
            st.line_chart(days.value_counts().sort_index())
        st.subheader("Latest scans")
        st.dataframe(det.head(5), use_container_width=True, hide_index=True)

elif page == "Detect":
    st.title("Detect wildlife")
    st.caption("Upload a photo of an animal. The app finds animals in the image, "
               "then names the species of each one.")

    col_a, col_b = st.columns([2, 1])
    with col_b:
        conf = st.slider("Minimum confidence", 0.01, 0.60, 0.10, 0.01,
                         help="Lower finds more animals but gives more false alarms.")
    with col_a:
        uploaded = st.file_uploader("Upload an animal image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image, err = read_image(uploaded)
        if err:
            st.error(err)
        else:
            if st.button("Run detection", type="primary"):
                try:
                    with st.spinner("Analysing image..."):
                        rows, annotated, note, debug = run_detection(image, conf)
                except Exception as e:
                    st.error("Detection could not run. Check that the model files "
                             "are in the project folder (first run needs internet).")
                    st.code(str(e))
                else:
                    if rows:
                        best = rows[0]
                        save_conf = best["Species confidence (%)"] or best["Detection confidence (%)"] or 0
                        save_detection(", ".join(dict.fromkeys(r["Animal"] for r in rows)),
                                       save_conf, "Animal detected")
                    else:
                        save_detection("No animal detected", 0, "No animal detected")
                    st.session_state.last_detection = {
                        "rows": rows, "annotated": annotated, "note": note, "debug": debug,
                        "time": datetime.now().strftime(TIME_FMT),
                    }

    res = st.session_state.last_detection
    if res:
        st.divider()
        img_col, info_col = st.columns([3, 2])
        with img_col:
            st.image(res["annotated"], caption="Detection result", use_container_width=True)
        with info_col:
            if res["rows"]:
                best = res["rows"][0]
                if best["Animal"] == "Unidentified animal":
                    st.warning("An animal was found, but the species is unclear.")
                else:
                    st.success(f"Wildlife detected: {best['Animal']}")
                    kpi("Species confidence", f"{best['Species confidence (%)']}%")
                    if best["Species confidence (%)"] < 40:
                        st.warning("Low confidence. Verify this result before relying on it.")
                if res["note"]:
                    st.caption(res["note"])
                st.dataframe(pd.DataFrame(res["rows"]), hide_index=True,
                             use_container_width=True)
                st.info("Next: open **Alerts** to test the 2 km warning.")
            else:
                st.warning("No animal found. Try a clearer image or lower the minimum confidence.")
        with st.expander("Technical details"):
            st.write(f"Animal boxes found by the detector: {res['debug']['boxes_found']}")
            st.write("Whole-image classifier top guesses: " + ", ".join(
                f"{n} ({c}%)" for n, c in res["debug"]["whole_image_top3"]))

elif page == "Alerts":
    st.title("Wildlife alert")
    st.markdown('<span class="badge">SIMULATED DISTANCE</span>', unsafe_allow_html=True)
    st.caption("No GPS or sensor is connected. You set the distance by hand to demonstrate "
               "the alert logic.")

    last = st.session_state.last_detection
    if last and last["rows"]:
        st.write(f"Latest detection: **{last['rows'][0]['Animal']}** "
                 f"at {last['time']}")
    else:
        st.write("No detection in this session yet. You can still test the alert below.")

    distance = st.slider("Estimated animal distance (km)", 0.0, 5.0, 3.0, 0.1)
    if distance <= ALERT_KM:
        st.markdown(f'<div class="alert-box alert-on">🚨 <b>Wildlife alert.</b> '
                    f'Animal is within {ALERT_KM:g} km. Avoid the area.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-box alert-off">🟢 Animal is outside the '
                    f'{ALERT_KM:g} km alert zone.</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("Save alert"):
        if distance <= ALERT_KM:
            save_alert(distance)
            st.success("Alert saved.")
        else:
            st.warning("Distance is above 2 km, so no alert was saved.")

    st.subheader("Alert history")
    alerts = load_alerts()
    if len(alerts):
        st.dataframe(alerts, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts saved yet. Set the distance to 2 km or less and save one.")

elif page == "History":
    st.title("Detection history")
    det = load_detections()
    if det.empty:
        st.info("No scans yet. Open **Detect** to run the first one.")
    else:
        choice = st.radio("Show", ["All", "Animal detected", "No animal detected"],
                          horizontal=True)
        view = det if choice == "All" else det[det["Status"] == choice]
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", view.to_csv(index=False).encode("utf-8"),
                           "wildlife_detection_history.csv", "text/csv")

else:
    st.title("About")
    st.write("Wildlife Guardian AI detects selected wildlife in uploaded images and "
             "demonstrates a distance-based safety alert.")
    left, right = st.columns(2)
    with left:
        st.subheader("Working now")
        st.markdown("- Image upload with validation\n- YOLO-World finds animals, YOLO11 names the species\n"
                    "- Detection history in SQLite, CSV export\n- Simulated 2 km alert")
    with right:
        st.subheader("Not implemented yet")
        st.markdown("- Live CCTV or video\n- Real GPS distance\n- Mobile notifications\n"
                    "- IoT sensors, cloud deployment\n- Custom-trained wildlife model")
    st.caption("Uses pretrained models. The classifier knows 1000 general ImageNet classes, so some "
               "animals (for example giraffe or deer) may show as unidentified. "
               "Predictions can be wrong and need human verification.")
