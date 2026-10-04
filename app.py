import streamlit as st
st.markdown(
    """
    <div style="text-align:center;">
        <h1>🦌 Wildlife Guardian AI</h1>
        <p style="font-size:18px;">
            AI-Powered Wildlife Detection & Public Safety Monitoring System
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.info(
    "🟢 System Status: Active  |  "
    "🤖 AI Detection: Ready  |  "
    "🚨 Alert Monitoring: Ready"
)
st.markdown(
    """
    #### 🔄 System Workflow

    **📷 Upload Image**
    →
    **🤖 AI Wildlife Detection**
    →
    **💾 Save Detection**
    →
    **📍 Distance Monitoring**
    →
    **🚨 Safety Alert**
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    ### 🌿 About Wildlife Guardian AI

    Wildlife Guardian AI is an AI-powered prototype designed to
    detect wildlife from uploaded images and provide a simulated
    safety alert when an animal is estimated to be within a
    2 km safety zone.
    """,
    unsafe_allow_html=True
)
import pandas as pd
import sqlite3
import os
from PIL import Image
from ultralytics import YOLOWorld
from ultralytics import YOLO
from datetime import datetime

st.set_page_config(
    page_title="Wildlife Guardian AI",
    page_icon="🐘",
    layout="wide"
)
DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "wildlife_history.db"
)


def init_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_time TEXT NOT NULL,
            animal_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()

def init_alert_db():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS alert_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_time TEXT NOT NULL,
            distance_km REAL NOT NULL,
            alert_status TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_alert_db()
def load_history():
    conn = sqlite3.connect(DB_PATH)

    rows = conn.execute("""
        SELECT date_time, animal_name, confidence, status
        FROM detection_history
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return [
        {
            "Date & Time": row[0],
            "Result": row[1],
            "Confidence (%)": row[2],
            "Status": row[3]
        }
        for row in rows
    ]
st.title("🐘 Wildlife Guardian AI")
st.subheader("AI-Powered Wildlife Detection System")

st.write(
    "Upload wildlife images, detect supported animals "
    "and monitor detection history."
)

# Store scan history during the current session
if "detection_log" not in st.session_state:
    st.session_state.detection_log = []

# Load the AI model only once
@st.cache_resource
def load_model():
    model = YOLOWorld("yolov8s-worldv2.pt")

    model.set_classes([
    "lion animal",
    "deer animal",
    "leopard animal",
    "elephant animal",
    "bear animal",
    "tiger animal",
    "zebra animal",
    "giraffe animal"
])

    return model
@st.cache_resource
def load_classifier():
    return YOLO("yolo11n-cls.pt")
st.divider()

st.header("📷 Wildlife Image Detection")
st.divider()

st.subheader("🤖 AI Wildlife Detection")

st.write(
    "Upload a wildlife image and let the AI model identify "
    "the detected animal with its confidence score."
)
uploaded_file = st.file_uploader(
    "Upload an animal image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Wildlife Image",
        use_container_width=True
    )

    if st.button("🔍 Detect Animal", type="primary"):
        with st.spinner("AI is analysing the image..."):
            try:
                model = load_model()
                classifier = load_classifier()
                results = model.predict(image, conf=0.01)
                
                result = results[0]
                class_result = classifier(image, verbose=False)[0]
                animal_classes = {
                    "lion animal",
                    "deer animal",
                    "leopard animal",
                    "elephant animal",
                    "bear animal",
                    "tiger animal",
                    "zebra animal",
                    "giraffe animal"
                }

                detected = []

                for box in result.boxes:
                    class_id = int(box.cls[0])
                    name = model.names[class_id]

                    if any(animal in name.lower() for animal in [
                        "lion", "deer", "leopard", "elephant",
                        "bear", "tiger", "zebra", "giraffe"
                    ]):
                        classified_id = int(class_result.probs.top1)
                        classified_name = class_result.names[classified_id]
                        classified_confidence = float(
                            class_result.probs.top1conf
                        ) * 100

                        detected.append({
                            "Animal": classified_name.title(),
                            "Confidence (%)": round(
                                classified_confidence, 1
                            )
                        })

                st.image(
                    image,
                    caption="AI Detection Result",
                    use_container_width=True
                )
                if detected:
                    st.success("🚨 Wildlife detected!")

                    st.subheader("📊 Detection Summary")

                    for animal in detected:
                        col1, col2 = st.columns(2)

                        col1.metric(
                            "🐾 Animal",
                            animal["Animal"]
                        )

                        col2.metric(
                            "🎯 Confidence",
                            f"{animal['Confidence (%)']}%"
                        )

                        st.progress(
                            min(animal["Confidence (%)"] / 100, 1.0)
                        )

                    animal_names = ", ".join(
                        item["Animal"] for item in detected
                    )

                    best_confidence = max(
                        item["Confidence (%)"] for item in detected
                    )

                    status = "Animal detected"

                else:
                    st.warning(
                        "No supported animal detected. "
                        "Try another image."
                    )

                    animal_names = "No animal detected"
                    best_confidence = 0
                    status = "No animal detected"

                st.session_state.detection_log.append({
                    "Date & Time": datetime.now().strftime(
                        "%d-%m-%Y %I:%M:%S %p"
                    ),
                    "Result": animal_names,
                    "Confidence (%)": best_confidence,
                    "Status": status
                })

                conn = sqlite3.connect(DB_PATH)

                conn.execute("""
                    INSERT INTO detection_history
                    (date_time, animal_name, confidence, status)
                    VALUES (?, ?, ?, ?)
                """, (
                    datetime.now().strftime("%d-%m-%Y %I:%M:%S %p"),
                    animal_names,
                    best_confidence,
                    status
                ))

                conn.commit()
                conn.close()

            except Exception as error:
                st.error("Detection failed. Please check the error.")
                st.code(str(error))

# Monitoring dashboard
st.header("📊 Monitoring Dashboard")

history = load_history()
total_scans = len(history)
animal_scans = sum(
    1 for item in history
    if item["Status"] == "Animal detected"
)

col1, col2, col3 = st.columns(3)

col1.metric("Total Scans", total_scans)
col2.metric("Scans with Animals", animal_scans)
col3.metric("No Animal Detected", total_scans - animal_scans)

if history:
    st.subheader("🕒 Detection History")

    df = pd.DataFrame(history)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Detection History (CSV)",
        data=csv_data,
        file_name="wildlife_detection_history.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear Current Session"):
        st.session_state.detection_log = []
        st.rerun()

else:
    st.info(
        "No scan history yet. Upload an image and click "
        "'Detect Animal' to start monitoring."
    )

st.divider()

st.subheader("🚨 Wildlife Alert Simulator")
st.caption(
    "⚠️ Demo Mode — Distance is manually simulated for prototype testing."
)

st.write(
    "Demo only: manually set the estimated distance "
    "to test the alert system."
)

distance_km = st.slider(
    "Estimated animal distance (km)",
    min_value=0.0,
    max_value=5.0,
    value=3.0,
    step=0.1
)
if distance_km <= 2.0:
    st.error("🚨 WILDLIFE ALERT! Animal is within 2 km.")
    st.warning("Demo alert only — no real-world distance is measured.")
else:
    st.success("🟢 Animal is outside the 2 km alert zone.")

if st.button("💾 Save Current Alert"):
    if distance_km <= 2.0:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            """
            INSERT INTO alert_history
            (date_time, distance_km, alert_status)
            VALUES (?, ?, ?)
            """,
            (
                datetime.now().strftime("%d-%m-%Y %I:%M:%S %p"),
                distance_km,
                "Wildlife Alert"
            )
        )
        conn.commit()
        conn.close()
        st.success("Alert saved to database!")
    else:
        st.warning("Distance is above 2 km. No alert saved.")
st.divider()
st.subheader("📋 Wildlife Alert History")

conn = sqlite3.connect(DB_PATH)

alert_rows = conn.execute("""
    SELECT date_time, distance_km, alert_status
    FROM alert_history
    ORDER BY id DESC
""").fetchall()

conn.close()

if alert_rows:
    alert_df = pd.DataFrame(
        alert_rows,
        columns=["Date & Time", "Distance (km)", "Alert Status"]
    )
    st.dataframe(
        alert_df,
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No wildlife alerts recorded yet.")
st.caption("This is a simulated distance, not an actual sensor measurement.")
st.caption(
    "Wildlife Guardian AI | Mini Project Prototype | "
    "AI predictions require verification."
)