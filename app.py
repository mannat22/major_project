# ================= IMPORTS ================= #
from pymongo import MongoClient
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ================= PAGE CONFIG ================= #
st.set_page_config(page_title="Smart Health AI", page_icon="🧠", layout="wide")

# ================= UI ================= #
st.markdown("""
<style>

.stApp {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c92d2);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.card {
    background: rgba(255,255,255,0.10);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 25px;
    margin: 15px 0px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.3);
    border: 1px solid rgba(255,255,255,0.15);
}

h1, h2, h3 {
    color: white;
    font-family: Arial;
}

.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    font-size: 16px;
    font-weight: bold;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER ================= #
st.markdown("<h1 style='text-align:center;'>🧠 Smart Health AI System</h1>", unsafe_allow_html=True)

# ================= LOAD MODEL ================= #
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("Model not found!")
    st.stop()

# ================= MONGODB ================= #
MONGO_URI = "mongodb+srv://amin:admin123@cluster0.27iplaf.mongodb.net/?appName=Cluster0"

client = MongoClient(MONGO_URI)
db = client["health_ai"]
collection = db["predictions"]

# ================= SIDEBAR ================= #
st.sidebar.title("🧾 Patient Input")

name = st.sidebar.text_input("👤 Name")

age = st.sidebar.slider("Age", 18, 100, 30)
weight = st.sidebar.slider("Weight (kg)", 40, 150, 70)
height = st.sidebar.slider("Height (cm)", 140, 210, 170)
sleep = st.sidebar.slider("Sleep Hours", 3, 10, 7)

exercise = 1 if st.sidebar.selectbox("Exercise?", ["No", "Yes"]) == "Yes" else 0
sugar = st.sidebar.slider("Sugar Intake", 0, 10, 5)
smoking = 1 if st.sidebar.selectbox("Smoking?", ["No", "Yes"]) == "Yes" else 0
alcohol = 1 if st.sidebar.selectbox("Alcohol?", ["No", "Yes"]) == "Yes" else 0

profession_list = ["Student","Engineer","Doctor","Teacher","Business","Other"]
profession = profession_list.index(st.sidebar.selectbox("Profession", profession_list))

bmi = weight / ((height/100)**2)

# ================= AI FUNCTIONS ================= #

def get_health_score(pred, bmi, sleep, sugar, smoking, alcohol):
    score = 100

    if pred == 1:
        score -= 35
    if bmi > 25:
        score -= 15
    if sleep < 6:
        score -= 10
    if sugar > 7:
        score -= 10
    if smoking:
        score -= 15
    if alcohol:
        score -= 10

    if score >= 75:
        level = "🟢 LOW RISK"
    elif score >= 50:
        level = "🟡 MODERATE RISK"
    else:
        level = "🔴 HIGH RISK"

    return score, level


def ai_recommendations(bmi, sleep, sugar, smoking, alcohol):
    tips = []

    if bmi > 25:
        tips.append("🥗 Reduce weight with healthy diet + exercise")
    if bmi < 18:
        tips.append("🍗 Increase nutritious food intake")

    if sleep < 6:
        tips.append("😴 Improve sleep (7–8 hours needed)")
    if sugar > 7:
        tips.append("🚫 Reduce sugar intake")
    if smoking:
        tips.append("🚭 Quit smoking gradually")
    if alcohol:
        tips.append("🍺 Limit alcohol consumption")

    if not tips:
        tips.append("✅ Maintain healthy lifestyle")

    return tips


def generate_pdf(name, score, level):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [
        Paragraph(f"Name: {name}", styles['Normal']),
        Paragraph(f"Health Score: {score}/100", styles['Normal']),
        Paragraph(f"Risk Level: {level}", styles['Normal']),
    ]

    doc.build(content)
    buffer.seek(0)
    return buffer

# ================= MAIN ================= #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Patient Overview")
st.write(f"Name: {name}")
st.write(f"Age: {age} | BMI: {bmi:.2f}")
st.markdown('</div>', unsafe_allow_html=True)

# ================= ANALYZE ================= #
if st.button("🔍 Analyze Health Risk"):

    if name.strip() == "":
        st.warning("Enter name first!")
        st.stop()

    input_data = pd.DataFrame([{
        "age": age,
        "weight": weight,
        "height": height,
        "exercise": exercise,
        "sleep": sleep,
        "sugar_intake": sugar,
        "smoking": smoking,
        "alcohol": alcohol,
        "profession": profession,
        "bmi": bmi
    }])

    input_scaled = scaler.transform(input_data)

    prediction = int(model.predict(input_scaled)[0])
    confidence = float(np.max(model.predict_proba(input_scaled)))

    # ================= AI ================= #
    score, level = get_health_score(
        prediction, bmi, sleep, sugar, smoking, alcohol
    )

    st.markdown(f"""
    <div class="card">
        <h2>Health Score: {score}/100</h2>
        <h2>{level}</h2>
        <h3>Confidence: {confidence*100:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    st.progress(score)

    # ================= RECOMMENDATIONS ================= #
    st.markdown("### 🧠 AI Recommendations")
    for tip in ai_recommendations(bmi, sleep, sugar, smoking, alcohol):
        st.write("✔️", tip)

    # ================= SAVE TO MONGODB ================= #
    record = {
        "name": name,
        "age": age,
        "bmi": float(bmi),
        "prediction": prediction,
        "health_score": score,
        "risk_level": level,
        "confidence": confidence,
        "timestamp": datetime.now()
    }

    collection.insert_one(record)
    st.success("📦 Data saved successfully!")

    # ================= PDF ================= #
    pdf = generate_pdf(name, score, level)
    st.download_button("📥 Download Report", pdf, "report.pdf")
