# ================= IMPORTS ================= #
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import BytesIO
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ================= PAGE CONFIG ================= #
st.set_page_config(page_title="Smart Health AI", page_icon="🧠", layout="wide")

# ================= UI STYLE ================= #
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
    color: white;
}

h1, h2, h3 {
    color: white;
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
    st.error("❌ Model not found!")
    st.stop()

# ================= FEATURES ================= #
FEATURES = [
    "age","weight","height","exercise","sleep",
    "sugar_intake","smoking","alcohol","profession","bmi"
]

# ================= SIDEBAR INPUT ================= #
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

def health_score(pred, bmi, sleep, sugar, smoking, alcohol):
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
        tips.append("🥗 Reduce oily food, start daily walking (30 min)")
    if bmi < 18:
        tips.append("🍗 Increase protein intake (eggs, milk, nuts)")
    if sleep < 6:
        tips.append("😴 Fix sleep schedule (7–8 hours daily)")
    if sugar > 7:
        tips.append("🚫 Reduce sugar and soft drinks")
    if smoking:
        tips.append("🚭 Quit smoking step-by-step")
    if alcohol:
        tips.append("🍺 Reduce alcohol consumption")

    if not tips:
        tips.append("✅ Maintain healthy lifestyle")

    return tips


# ================= PDF FUNCTION ================= #
def generate_pdf(name, score, level, confidence, recommendations):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Smart Health AI Report", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Name: {name}", styles["Normal"]))
    content.append(Paragraph(f"Health Score: {score}/100", styles["Normal"]))
    content.append(Paragraph(f"Risk Level: {level}", styles["Normal"]))
    content.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles["Normal"]))

    content.append(Spacer(1, 10))
    content.append(Paragraph("AI Recommendations:", styles["Heading2"]))

    for r in recommendations:
        content.append(Paragraph("• " + r, styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer


# ================= MAIN UI ================= #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Patient Overview")
st.write(f"👤 Name: {name}")
st.write(f"Age: {age} | BMI: {bmi:.2f}")
st.markdown('</div>', unsafe_allow_html=True)

# ================= PREDICTION ================= #
if st.button("🔍 Analyze Health Risk"):

    if name.strip() == "":
        st.warning("Please enter your name!")
        st.stop()

    input_data = pd.DataFrame([[
        age, weight, height, exercise, sleep,
        sugar, smoking, alcohol, profession, bmi
    ]], columns=FEATURES)

    input_scaled = scaler.transform(input_data)

    prediction = int(model.predict(input_scaled)[0])
    confidence = float(np.max(model.predict_proba(input_scaled)))

    score, level = health_score(prediction, bmi, sleep, sugar, smoking, alcohol)

    recommendations = ai_recommendations(bmi, sleep, sugar, smoking, alcohol)

    # ================= RESULT ================= #
    st.markdown(f"""
    <div class="card">
        <h2>Health Score: {score}/100</h2>
        <h2>{level}</h2>
        <h3>Confidence: {confidence*100:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    st.progress(score)

    # ================= RECOMMENDATIONS ================= #
    st.markdown("### 🧠 AI Lifestyle Recommendations")
    for r in recommendations:
        st.write("✔️", r)

    # ================= PDF DOWNLOAD ================= #
    pdf = generate_pdf(name, score, level, confidence, recommendations)

    st.download_button(
        "📥 Download Health Report (PDF)",
        pdf,
        file_name="health_report.pdf",
        mime="application/pdf"
    )
