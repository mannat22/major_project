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

# ================= UI CONFIG ================= #
st.set_page_config(page_title="Smart Health AI", page_icon="🏥", layout="wide")

# ================= CUSTOM CSS ================= #
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
}
.block-container {
    padding: 2rem 3rem;
}
.card {
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    color: white;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.2);
    margin-bottom: 20px;
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}
section[data-testid="stSidebar"] {
    background: #1f2937;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER ================= #
st.markdown("""
<h1 style='text-align:center; color:white; font-size:50px;'>
🏥 Smart Health Risk AI
</h1>
<p style='text-align:center; color:#ddd; font-size:18px;'>
AI-powered health prediction with personalized insights
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

profession_list = [
    "Student","Software Engineer","Doctor","Teacher","Business",
    "Engineer","Government Job","Self-Employed","Unemployed","Other"
]
profession = profession_list.index(st.sidebar.selectbox("Profession", profession_list))

# BMI
bmi = weight / ((height/100)**2)

# ================= HELPER FUNCTIONS ================= #
def get_health_advice(prediction, age, bmi, smoking, alcohol, sleep, exercise, sugar):
    advice, diet, lifestyle, remedies = [], [], [], []

    if prediction == 1:
        advice.append("⚠️ High health risk detected. Improve lifestyle immediately.")
    else:
        advice.append("✅ Low health risk. Maintain your healthy routine.")

    if bmi >= 25:
        lifestyle.append("⚖️ Overweight detected.")
        diet.append("🥗 Eat healthy low-calorie food.")
    elif bmi < 18.5:
        lifestyle.append("⚠️ Underweight.")
        diet.append("🥛 Increase protein intake.")

    if sleep < 6:
        lifestyle.append("😴 Improve sleep schedule.")

    if sugar > 7:
        diet.append("🍬 Reduce sugar intake.")

    if smoking:
        lifestyle.append("🚭 Quit smoking.")

    if alcohol:
        lifestyle.append("🍺 Reduce alcohol.")

    if exercise == 0:
        lifestyle.append("🏃 Start daily exercise.")

    remedies += ["🌿 Lemon water", "🌿 Turmeric milk", "🌿 Stay hydrated"]

    return advice, diet, lifestyle, remedies


def generate_pdf(age, weight, height, sleep, sugar, bmi, prediction, confidence,
                 advice, diet, lifestyle, remedies):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Smart Health Report", styles['Title']))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Age: {age}", styles['Normal']))
    content.append(Paragraph(f"BMI: {bmi:.2f}", styles['Normal']))
    content.append(Paragraph(f"Prediction: {'High Risk' if prediction==1 else 'Low Risk'}", styles['Normal']))
    content.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles['Normal']))

    for section in [advice, diet, lifestyle, remedies]:
        content.append(Spacer(1, 10))
        for item in section:
            content.append(Paragraph(item, styles['Normal']))

    doc.build(content)
    buffer.seek(0)
    return buffer

# ================= MAIN ================= #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Patient Overview")
st.write(f"Age: {age} | BMI: {bmi:.2f} | Sleep: {sleep} hrs")
st.markdown('</div>', unsafe_allow_html=True)

analyze = st.button("🔍 Analyze Health Risk")

if analyze:

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
        <h2 style='text-align:center;'>{"⚠️ HIGH RISK" if prediction==1 else "✅ LOW RISK"}</h2>
        <h3 style='text-align:center;'>Confidence: {confidence*100:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    # GRAPH
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Confidence Analysis")

    fig, ax = plt.subplots()
    ax.bar(["Confidence"], [confidence*100])
    st.pyplot(fig)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    # SUGGESTIONS
    advice, diet, lifestyle, remedies = get_health_advice(
        prediction, age, bmi, smoking, alcohol, sleep, exercise, sugar
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🩺 Health Suggestions")

    for section, title in zip([advice, diet, lifestyle, remedies],
                             ["Summary", "Diet", "Lifestyle", "Remedies"]):
        st.write(f"### {title}")
        for item in section:
            st.write(item)

    st.markdown('</div>', unsafe_allow_html=True)

    # SAVE CSV
    record = pd.DataFrame([{
        "age": age, "bmi": bmi, "prediction": prediction, "confidence": confidence
    }])

    if os.path.exists("data.csv"):
        record.to_csv("data.csv", mode='a', header=False, index=False)
    else:
        record.to_csv("data.csv", index=False)

    # PDF
    pdf = generate_pdf(age, weight, height, sleep, sugar, bmi,
                       prediction, confidence,
                       advice, diet, lifestyle, remedies)

    st.download_button("📥 Download Report", pdf, "health_report.pdf")

# HISTORY
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📂 Previous Records")

if os.path.exists("data.csv"):
    df = pd.read_csv("data.csv")
    st.dataframe(df.tail(5))
else:
    st.write("No records yet.")

st.markdown('</div>', unsafe_allow_html=True)
