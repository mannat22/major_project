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

# ================= UI CSS ================= #
st.markdown("""
<style>
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
.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    color: white;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    margin-bottom: 20px;
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER ================= #
st.markdown("""
<h1 style='text-align:center; color:white;'>🧠 Smart Health AI</h1>
""", unsafe_allow_html=True)

# ================= LOAD MODEL ================= #
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("Model not found!")
    st.stop()

# ================= SIDEBAR ================= #
st.sidebar.title("🧾 Patient Input")

name = st.sidebar.text_input("👤 Enter Name")

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
def generate_pdf(name, prediction, confidence):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph(f"Name: {name}", styles['Normal']))
    content.append(Paragraph(f"Prediction: {'High Risk' if prediction else 'Low Risk'}", styles['Normal']))
    content.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles['Normal']))

    doc.build(content)
    buffer.seek(0)
    return buffer

# ================= MAIN ================= #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Patient Overview")
st.write(f"👤 Name: {name}")
st.write(f"Age: {age} | BMI: {bmi:.2f}")
st.markdown('</div>', unsafe_allow_html=True)

if st.button("🔍 Analyze Health Risk"):

    if name.strip() == "":
        st.warning("Please enter your name!")
        st.stop()

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

    # ================= RESULT ================= #
    st.markdown(f"""
    <div class="card">
        <h2>{'⚠️ HIGH RISK' if prediction else '✅ LOW RISK'}</h2>
        <h3>Confidence: {confidence*100:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(confidence * 100))

    # ================= SAVE DATA ================= #
    file_path = "data.csv"

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        serial = len(df) + 1
    else:
        serial = 1

    new_record = pd.DataFrame([{
        "S.No": serial,
        "Name": name,
        "Age": age,
        "BMI": bmi,
        "Prediction": prediction,
        "Confidence": confidence
    }])

    if os.path.exists(file_path):
        new_record.to_csv(file_path, mode='a', header=False, index=False)
    else:
        new_record.to_csv(file_path, index=False)

    # ================= PDF ================= #
    pdf = generate_pdf(name, prediction, confidence)

    st.download_button("📥 Download Report", pdf, "report.pdf")



import os

st.write("Model exists:", os.path.exists("model.pkl"))
st.write("Scaler exists:", os.path.exists("scaler.pkl"))
