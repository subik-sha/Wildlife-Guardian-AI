# 🦌 Wildlife Guardian AI

### AI-Powered Wildlife Detection & Public Safety Monitoring System

Wildlife Guardian AI is a prototype system designed to detect wildlife and support public safety in areas near forests, wildlife zones, and roads where human-wildlife encounters may occur.

## 🎯 Problem

People travelling through forest-side roads and wildlife-prone areas may encounter animals unexpectedly. Early detection can help improve awareness and safety.

## 💡 Solution

Wildlife Guardian AI uses AI-based image detection and classification to identify wildlife from uploaded images and provides a safety alert when the simulated animal distance is within 2 km.

## ✨ Features

- 🤖 AI-based wildlife detection
- 🐾 Wildlife classification
- 📊 Detection confidence display
- 💾 SQLite detection history
- 📈 Monitoring dashboard
- 🚨 2 km distance-based alert simulation
- 📥 Detection history export
- 🌐 Streamlit web interface

## 🛠️ Technologies Used

- Python
- Streamlit
- YOLO
- YOLO-World
- SQLite
- OpenCV
- Pandas
- Pillow

## 🔄 System Workflow

```text
📷 Upload Image
      ↓
🤖 AI Wildlife Detection
      ↓
🐾 Animal Classification
      ↓
📊 Confidence Analysis
      ↓
💾 Save Detection History
      ↓
📍 Distance Monitoring
      ↓
🚨 Safety Alert