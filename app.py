# ================= IMPORTS ================= #
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# ================= PAGE CONFIG ================= #
st.set_page_config(page_title="Smart Health AI", page_icon="🧠", layout="wide")

# ================= ADVANCED CSS ================= #
st.markdown("""
<style>

/* Background Gradient Animation */
body {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #00c6ff);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Glass Card */
.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.02);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 18px;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
    color: white;
}

/* Glow Text */
.glow {
    color: #fff;
    text-shadow: 0 0 10px #00c6ff, 0 0 20px #00c6ff;
}

/* Progress Bar Animation */
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #00c6ff, #0072ff);
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER ================= #
st.markdown("""
<h1 class="glow" style='text-align:center; font-size:55px;'>
🧠 Smart Health AI
</h1>
<p style='text-align:center; color:#ddd; font-size:18px;'>
Next-Gen AI Health Prediction System
</p>
""", unsafe_allow_html=True)

# ================= LOAD MODEL ================= #
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    st.error("Model or Scaler file not found.")
    st.stop()

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# ================= SIDEBAR ================= #
st.sidebar.title("🧾 Patient Input")

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

bmi = weight / ((height/100)**2)

# ================= FUNCTIONS ================= #
def get_health_advice(prediction):
    return ["Stay active", "Eat healthy", "Sleep well"]

def generate_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    content = [Paragraph("Health Report", styles['Title'])]
    doc.build(content)
    buffer.seek(0)
    return buffer

# ================= MAIN CARD ================= #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Patient Overview")
st.write(f"Age: {age} | BMI: {bmi:.2f} | Sleep: {sleep} hrs")
st.markdown('</div>', unsafe_allow_html=True)

# ================= BUTTON ================= #
if st.button("🔍 Analyze Health Risk"):

    input_data = pd.DataFrame([{
        "age": age, "weight": weight, "height": height,
        "exercise": exercise, "sleep": sleep,
        "sugar_intake": sugar, "smoking": smoking,
        "alcohol": alcohol, "married": married,
        "profession": profession, "bmi": bmi
    }])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    confidence = float(np.max(model.predict_proba(input_scaled)))

    # RESULT CARD
    st.markdown(f"""
    <div class="card">
        <h2 style='text-align:center;'>{'⚠️ HIGH RISK' if prediction else '✅ LOW RISK'}</h2>
        <h3 style='text-align:center;'>Confidence: {confidence*100:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    # ANIMATED PROGRESS
    st.progress(int(confidence * 100))

    # CHART
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    ax.bar(["Confidence"], [confidence*100])
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # SUGGESTIONS
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🩺 Suggestions")
    for tip in get_health_advice(prediction):
        st.write("✔️", tip)
    st.markdown('</div>', unsafe_allow_html=True)

    # PDF
    pdf = generate_pdf()
    st.download_button("📥 Download Report", pdf, "report.pdf")

# ================= FOOTER ================= #
st.markdown("""
<p style='text-align:center; color:#aaa; margin-top:30px;'>
Made with ❤️ using AI
</p>
""", unsafe_allow_html=True)
