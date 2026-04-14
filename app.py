# ================= IMPORTS ================= #
import streamlit as st
import numpy as np
import joblib
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

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
married = 1 if st.sidebar.selectbox("Married?", ["No", "Yes"]) == "Yes" else 0

profession_list = ["Student","Engineer","Doctor","Teacher","Business","Other"]
profession = profession_list.index(st.sidebar.selectbox("Profession", profession_list))

# ================= BMI ================= #
bmi = weight / ((height / 100) ** 2)

# ================= FEATURE ORDER (CRITICAL) ================= #
FEATURES = [
    "age",
    "weight",
    "height",
    "exercise",
    "sleep",
    "sugar_intake",
    "smoking",
    "alcohol",
    "married",
    "profession",
    "bmi"
]

# ================= AI LOGIC ================= #
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


def recommendations(bmi, sleep, sugar, smoking, alcohol):
    tips = []

    if bmi > 25:
        tips.append("🥗 Reduce oily food & start walking 30 min daily")
    if bmi < 18:
        tips.append("🍗 Increase protein intake")
    if sleep < 6:
        tips.append("😴 Improve sleep cycle (7–8 hrs)")
    if sugar > 7:
        tips.append("🚫 Reduce sugar & soft drinks")
    if smoking:
        tips.append("🚭 Quit smoking gradually")
    if alcohol:
        tips.append("🍺 Reduce alcohol intake")

    if not tips:
        tips.append("✅ Maintain healthy lifestyle")

    return tips


# ================= PDF ================= #
def generate_pdf(name, score, level, confidence, recs):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [
        Paragraph("Smart Health AI Report", styles["Title"]),
        Spacer(1, 10),
        Paragraph(f"Name: {name}", styles["Normal"]),
        Paragraph(f"Health Score: {score}/100", styles["Normal"]),
        Paragraph(f"Risk Level: {level}", styles["Normal"]),
        Paragraph(f"Confidence: {confidence*100:.2f}%", styles["Normal"]),
        Spacer(1, 10),
        Paragraph("Recommendations:", styles["Heading2"]),
    ]

    for r in recs:
        content.append(Paragraph("• " + r, styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer


# ================= MAIN ================= #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Patient Overview")
st.write(f"👤 Name: {name}")
st.write(f"Age: {age} | BMI: {bmi:.2f}")
st.markdown('</div>', unsafe_allow_html=True)

# ================= PREDICTION ================= #
if st.button("🔍 Analyze Health Risk"):

    if name.strip() == "":
        st.warning("Please enter name!")
        st.stop()

    # ✅ FIXED INPUT (NO ERROR GUARANTEE)
    input_array = np.array([[
        age,
        weight,
        height,
        exercise,
        sleep,
        sugar,
        smoking,
        alcohol,
        married,
        profession,
        bmi
    ]])

    input_scaled = scaler.transform(input_array)

    prediction = int(model.predict(input_scaled)[0])
    confidence = float(np.max(model.predict_proba(input_scaled)))

    score, level = health_score(prediction, bmi, sleep, sugar, smoking, alcohol)
    recs = recommendations(bmi, sleep, sugar, smoking, alcohol)

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
    st.markdown("### 🧠 AI Recommendations")
    for r in recs:
        st.write("✔️", r)

    # ================= PDF DOWNLOAD ================= #
    pdf = generate_pdf(name, score, level, confidence, recs)

    st.download_button(
        "📥 Download Health Report",
        pdf,
        file_name="health_report.pdf",
        mime="application/pdf"
    )
