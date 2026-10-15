"""Wildlife Guardian AI - Hackathon Prototype.
Animal detection and species classification from uploaded images.
Distance alerts are simulated; no GPS or IoT sensor is connected.
"""

import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


st.set_page_config(
    page_title="Wildlife Guardian AI",
    page_icon="🐘",
    layout="wide",
)

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "wildlife_history.db",
)

TIME_FMT = "%d-%m-%Y %I:%M:%S %p"
ALERT_KM = 2.0
MAX_UPLOAD_MB = 10

WILDLIFE_WORDS = (
    "lion", "tiger", "leopard", "jaguar", "cheetah",
    "elephant", "tusker", "zebra", "bear", "hippopotamus",
    "impala", "gazelle", "hartebeest", "ibex", "bighorn",
    "warthog", "boar", "buffalo", "giraffe", "deer",
    "wolf", "fox", "hyena", "monkey", "gorilla",
    "chimpanzee", "baboon", "rhinoceros", "antelope",
    "coyote", "dingo",
)


# ---------- Styling ----------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f4f8f1, #e7f0e5, #f8f5e9);
    color: #20382d;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
h1, h2, h3 {
    font-family: Georgia, 'Times New Roman', serif;
    color: #214d37;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #173e2c, #28583d);
}
[data-testid="stSidebar"] * {
    color: #f2f6ed;
}
.kpi {
    background: #fff;
    border: 1px solid #d6e5d7;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 14px rgba(32,72,48,.08);
}
.kpi .label {
    color: #587463;
    font-size: .85rem;
}
.kpi .value {
    font-family: Georgia, serif;
    font-size: 2rem;
    line-height: 1.3;
    color: #214d37;
    overflow-wrap: anywhere;
}
.kpi.warn {
    border-left: 4px solid #d6a23a;
}
.stButton > button, .stDownloadButton > button {
    border-radius: 9px;
    border: 1px solid #2f704a;
}
.stButton > button[kind="primary"] {
    background: #286441;
    color: white;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,.7);
    border: 1px dashed #9dbda4;
    border-radius: 12px;
    padding: 10px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid #d6e5d7;
    border-radius: 10px;
    overflow: hidden;
}
.badge {
    display: inline-block;
    padding: 4px 11px;
    border-radius: 20px;
    font-size: .78rem;
    border: 1px solid #d6a23a;
    color: #855c12;
    background: #fff8e5;
}
.alert-box {
    border-radius: 12px;
    padding: 18px 20px;
    font-size: 1.05rem;
}
.alert-on {
    background: #fff0e9;
    border: 1px solid #d1603d;
    color: #8b3020;
}
.alert-off {
    background: #e8f5eb;
    border: 1px solid #77ae87;
    color: #245a39;
}
[data-testid="stImage"] img {
    border-radius: 12px;
}
@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
    }
    .kpi .value {
        font-size: 1.5rem;
    }
}
</style>
""", unsafe_allow_html=True)


def kpi(label, value, warn=False):
    cls = "kpi warn" if warn else "kpi"
    st.markdown(
        f'<div class="{cls}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )


# ---------- Database ----------

def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_time TEXT NOT NULL,
                animal_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_time TEXT NOT NULL,
                distance_km REAL NOT NULL,
                alert_status TEXT NOT NULL
            )
        """)


init_db()


def load_detections():
    with db() as conn:
        rows = conn.execute("""
            SELECT date_time, animal_name, confidence, status
            FROM detection_history
            ORDER BY id DESC
        """).fetchall()

    return pd.DataFrame(
        rows,
        columns=[
            "Date & Time",
            "Result",
            "Confidence (%)",
            "Status",
        ],
    )


def load_alerts():
    with db() as conn:
        rows = conn.execute("""
            SELECT date_time, distance_km, alert_status
            FROM alert_history
            ORDER BY id DESC
        """).fetchall()

    return pd.DataFrame(
        rows,
        columns=["Date & Time", "Distance (km)", "Alert Status"],
    )


def save_detection(result, confidence, status):
    with db() as conn:
        conn.execute("""
            INSERT INTO detection_history
            (date_time, animal_name, confidence, status)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now().strftime(TIME_FMT),
            result,
            float(confidence),
            status,
        ))


def save_alert(distance_km):
    with db() as conn:
        conn.execute("""
            INSERT INTO alert_history
            (date_time, distance_km, alert_status)
            VALUES (?, ?, ?)
        """, (
            datetime.now().strftime(TIME_FMT),
            float(distance_km),
            "Wildlife Alert (simulated)",
        ))


# ---------- AI Models ----------

@st.cache_resource(show_spinner="Loading detection model...")
def load_detector():
    from ultralytics import YOLOWorld

    model = YOLOWorld("yolov8s-worldv2.pt")
    model.set_classes(["animal"])
    return model


@st.cache_resource(show_spinner="Loading classifier...")
def load_classifier():
    from ultralytics import YOLO

    return YOLO("yolo11n-cls.pt")


def nice(name):
    return name.replace("_", " ").title()


def is_wildlife(name):
    name = name.lower().replace("_", " ")
    return any(word in name for word in WILDLIFE_WORDS)


def classify(img):
    result = load_classifier()(img, verbose=False)[0]
    ids = [int(i) for i in result.probs.top5]
    confs = [float(c) for c in result.probs.top5conf]

    return [
        (result.names[i], confidence)
        for i, confidence in zip(ids, confs)
    ]


def read_image(uploaded):
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"File is larger than {MAX_UPLOAD_MB} MB."

    try:
        img = Image.open(uploaded)
        img.load()
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None, "Unable to read image. Upload a JPG or PNG."

    img.thumbnail((1280, 1280))
    return img, None


def draw_boxes(image, items):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    size = max(16, output.width // 40)

    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), label in items:
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(111, 165, 138),
            width=max(3, output.width // 250),
        )

        text_width = draw.textlength(label, font=font)
        top = max(0, y1 - size - 6)

        draw.rectangle(
            [x1, top, x1 + text_width + 10, top + size + 6],
            fill=(111, 165, 138),
        )
        draw.text(
            (x1 + 5, top + 2),
            label,
            fill=(16, 26, 22),
            font=font,
        )

    return output


def run_detection(image, conf):
    detector_result = load_detector().predict(
        image,
        conf=conf,
        verbose=False,
    )[0]

    boxes = sorted(
        detector_result.boxes,
        key=lambda box: float(box.conf[0]),
        reverse=True,
    )[:5]

    width, height = image.size
    rows = []
    drawn = []

    for box in boxes:
        x1, y1, x2, y2 = [
            int(value) for value in box.xyxy[0].tolist()
        ]

        pad = int(0.05 * max(x2 - x1, y2 - y1))

        crop = image.crop((
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(width, x2 + pad),
            min(height, y2 + pad),
        ))

        name, confidence = classify(crop)[0]

        if is_wildlife(name) and confidence >= 0.25:
            label = nice(name)
            species_conf = round(confidence * 100, 1)
        else:
            label = "Unidentified animal"
            species_conf = 0.0

        rows.append({
            "Animal": label,
            "Species confidence (%)": species_conf,
            "Detection confidence (%)": round(
                float(box.conf[0]) * 100, 1
            ),
        })

        drawn.append(((x1, y1, x2, y2), label))

    whole = classify(image)
    note = ""

    if not rows:
        name, confidence = whole[0]

        if is_wildlife(name) and confidence >= 0.30:
            rows.append({
                "Animal": nice(name),
                "Species confidence (%)": round(confidence * 100, 1),
                "Detection confidence (%)": None,
            })
            note = (
                "No detection box was found. "
                "The whole image was classified instead."
            )

    rows.sort(
        key=lambda row: (
            row["Species confidence (%)"],
            row["Detection confidence (%)"] or 0,
        ),
        reverse=True,
    )

    debug = {
        "boxes_found": len(boxes),
        "whole_image_top3": [
            (nice(name), round(score * 100, 1))
            for name, score in whole[:3]
        ],
    }

    return rows, draw_boxes(image, drawn), note, debug


# ---------- Navigation ----------

st.sidebar.title("🐘 Wildlife Guardian")
st.sidebar.caption("AI wildlife detection and safety alerts")

page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Detect", "Alerts", "History", "About"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption(
    "Hackathon prototype: uploaded-image detection; distance is simulated."
)

if "last_detection" not in st.session_state:
    st.session_state.last_detection = None


# ---------- Dashboard ----------

if page == "Dashboard":
    st.title("Monitoring dashboard")

    detections = load_detections()
    alerts = load_alerts()

    total = len(detections)
    with_animals = (
        int((detections["Status"] == "Animal detected").sum())
        if total else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi("Total scans", total)
    with c2:
        kpi("Scans with animals", with_animals)
    with c3:
        kpi("No animal found", total - with_animals)
    with c4:
        kpi("Alerts saved (simulated)", len(alerts), warn=True)

    st.write("")

    if total == 0:
        st.info("No scans yet. Open Detect and upload an image.")
    else:
        left, right = st.columns(2)

        found = detections[
            detections["Status"] == "Animal detected"
        ]

        with left:
            st.subheader("Detections by animal")

            if len(found):
                counts = (
                    found["Result"]
                    .str.replace(
                        " animal", "",
                        case=False,
                        regex=False,
                    )
                    .str.title()
                    .str.split(", ")
                    .explode()
                    .value_counts()
                )
                st.bar_chart(counts)
            else:
                st.caption("No animals detected yet.")

        with right:
            st.subheader("Scans per day")

            days = pd.to_datetime(
                detections["Date & Time"],
                format=TIME_FMT,
                errors="coerce",
            ).dt.date

            st.line_chart(days.value_counts().sort_index())

        st.subheader("Latest scans")
        st.dataframe(
            detections.head(5),
            use_container_width=True,
            hide_index=True,
        )


# ---------- Detect ----------

elif page == "Detect":
    st.title("Detect wildlife")
    st.caption(
        "Upload a photo. The app finds animal regions and attempts "
        "to classify their species."
    )

    col_a, col_b = st.columns([2, 1])

    with col_b:
        conf = st.slider(
            "Minimum confidence",
            0.01, 0.60, 0.10, 0.01,
            help="Lower values may find more animals but can increase false alarms.",
        )

    with col_a:
        uploaded = st.file_uploader(
            "Upload an animal image",
            type=["jpg", "jpeg", "png"],
        )

    if uploaded is not None:
        image, error = read_image(uploaded)

        if error:
            st.error(error)
        else:
            st.image(
                image,
                caption="Uploaded image",
                use_container_width=True,
            )

            if st.button("Run detection", type="primary"):
                try:
                    with st.spinner("Analysing image..."):
                        rows, annotated, note, debug = run_detection(
                            image, conf
                        )
                except Exception as exc:
                    st.error(
                        "Detection failed. Check internet access, "
                        "model downloads and installed dependencies."
                    )
                    st.code(str(exc))
                else:
                    if rows:
                        best = rows[0]
                        save_conf = (
                            best["Species confidence (%)"]
                            or best["Detection confidence (%)"]
                            or 0
                        )

                        save_detection(
                            ", ".join(dict.fromkeys(
                                row["Animal"] for row in rows
                            )),
                            save_conf,
                            "Animal detected",
                        )
                    else:
                        save_detection(
                            "No animal detected",
                            0,
                            "No animal detected",
                        )

                    st.session_state.last_detection = {
                        "rows": rows,
                        "annotated": annotated,
                        "note": note,
                        "debug": debug,
                        "time": datetime.now().strftime(TIME_FMT),
                    }

    result = st.session_state.last_detection

    if result:
        st.divider()
        image_col, info_col = st.columns([3, 2])

        with image_col:
            st.image(
                result["annotated"],
                caption="Detection result",
                use_container_width=True,
            )

        with info_col:
            if result["rows"]:
                best = result["rows"][0]

                if best["Animal"] == "Unidentified animal":
                    st.warning(
                        "An animal was found, but its species is unclear."
                    )
                else:
                    st.success(
                        f"Wildlife detected: {best['Animal']}"
                    )
                    kpi(
                        "Species confidence",
                        f"{best['Species confidence (%)']}%",
                    )

                    if best["Species confidence (%)"] < 40:
                        st.warning(
                            "Low confidence. Verify this result."
                        )

                if result["note"]:
                    st.caption(result["note"])

                st.dataframe(
                    pd.DataFrame(result["rows"]),
                    hide_index=True,
                    use_container_width=True,
                )
                st.info(
                    "Open Alerts to test the simulated 2 km warning."
                )
            else:
                st.warning(
                    "No animal found. Try a clearer image or adjust confidence."
                )

        with st.expander("Technical details"):
            st.write(
                "Animal boxes found: "
                f"{result['debug']['boxes_found']}"
            )
            st.write(
                "Whole-image classifier top guesses: "
                + ", ".join(
                    f"{name} ({score}%)"
                    for name, score
                    in result["debug"]["whole_image_top3"]
                )
            )


# ---------- Alerts ----------

elif page == "Alerts":
    st.title("Wildlife alert")

    st.markdown(
        '<span class="badge">SIMULATED DISTANCE</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "No GPS or sensor is connected. Set the distance manually "
        "to demonstrate the alert logic."
    )

    last = st.session_state.last_detection

    if last and last["rows"]:
        st.write(
            f"Latest detection: **{last['rows'][0]['Animal']}** "
            f"at {last['time']}"
        )
    else:
        st.write(
            "No detection in this session yet. "
            "You can still test the alert below."
        )

    distance = st.slider(
        "Estimated animal distance (km)",
        0.0, 5.0, 3.0, 0.1,
    )

    if distance <= ALERT_KM:
        st.markdown(
            f'<div class="alert-box alert-on">🚨 '
            f'<b>Wildlife alert.</b> Animal is within '
            f'{ALERT_KM:g} km. Avoid the area.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="alert-box alert-off">🟢 Animal is outside '
            f'the {ALERT_KM:g} km alert zone.</div>',
            unsafe_allow_html=True,
        )

    if st.button("Save alert"):
        if distance <= ALERT_KM:
            save_alert(distance)
            st.success("Alert saved.")
        else:
            st.warning(
                "Distance is above 2 km, so no alert was saved."
            )

    st.subheader("Alert history")
    alerts = load_alerts()

    if len(alerts):
        st.dataframe(
            alerts,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No alerts saved yet. Set the distance to 2 km or less "
            "and save one."
        )


# ---------- History ----------

elif page == "History":
    st.title("Detection history")
    detections = load_detections()

    if detections.empty:
        st.info("No scans yet. Open Detect to run the first one.")
    else:
        choice = st.radio(
            "Show",
            ["All", "Animal detected", "No animal detected"],
            horizontal=True,
        )

        view = (
            detections
            if choice == "All"
            else detections[detections["Status"] == choice]
        )

        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download CSV",
            view.to_csv(index=False).encode("utf-8"),
            "wildlife_detection_history.csv",
            "text/csv",
        )


# ---------- About ----------

else:
    st.title("About Wildlife Guardian AI")

    st.write(
        "Wildlife Guardian AI is a hackathon prototype that detects "
        "animal-like regions in uploaded images and demonstrates a "
        "distance-based safety alert."
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Working in this prototype")
        st.markdown(
            "- Image upload and validation\n"
            "- YOLO-World animal-region detection\n"
            "- YOLO11 image classification attempt\n"
            "- SQLite detection and alert history\n"
            "- CSV export and simulated 2 km alert"
        )

    with right:
        st.subheader("Future development")
        st.markdown(
            "- Live CCTV/video input\n"
            "- GPS-based distance estimation\n"
            "- Mobile notifications\n"
            "- IoT sensors and field testing\n"
            "- Wildlife-specific model training"
        )

    st.caption(
        "This prototype uses pretrained models. Predictions may be "
        "incorrect or species may be unidentified. Do not rely on it "
        "for real-world safety decisions."
    )