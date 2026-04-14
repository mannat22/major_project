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
from pymongo import MongoClient
from datetime import datetime

# ================= MONGODB SETUP ================= #
MONGO_URI = "YOUR_MONGODB_ATLAS_URI"  # 🔥 Replace this

client = MongoClient(MONGO_URI)
db = client["health_ai"]
collection = db["predictions"]

# ================= AI SUGGESTIONS ================= #
def get_health_advice(prediction, age, bmi, smoking, alcohol, sleep, exercise, sugar):
    advice, diet, lifestyle, remedies = [], [], [], []

    if prediction == 1:
        advice.append("⚠️ High health risk detected. Improve lifestyle immediately.")
    else:
        advice.append("✅ Low health risk. Maintain your healthy routine.")

    if bmi >= 25:
        lifestyle.append("⚖️ Overweight detected. Focus on weight reduction.")
        diet.append("🥗 Eat salads, fruits, and low-calorie foods.")
        diet.append("🚫 Avoid fried and junk food.")
    elif bmi < 18.5:
        lifestyle.append("⚠️ Underweight. Increase nutritious food intake.")
        diet.append("🥛 Include milk, nuts, and proteins.")

    if sleep < 6:
        lifestyle.append("😴 Poor sleep. Aim for 7–8 hours daily.")

    if sugar > 7:
        diet.append("🍬 Reduce sugar intake immediately.")

    if smoking:
        lifestyle.append("🚭 Quit smoking gradually.")

    if alcohol:
        lifestyle.append("🍺 Limit alcohol consumption.")

    if exercise == 0:
        lifestyle.append("🏃 Start 20–30 minutes walking daily.")
    else:
        lifestyle.append("💪 Good physical activity level.")

    if age > 50:
        lifestyle.append("🩺 Regular health checkups recommended.")

    remedies.append("🌿 Drink warm lemon water in morning.")
    remedies.append("🌿 Take turmeric milk for immunity.")
    remedies.append("🌿 Drink 2–3L water daily.")

    return advice, diet, lifestyle, remedies

# ================= PDF FUNCTION ================= #
def generate_pdf(age, weight, height, sleep, sugar, bmi, prediction, confidence,
                 advice, diet, lifestyle, remedies):

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("Smart Health Risk Report", styles['Title']))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Age: {age}", styles['Normal']))
    content.append(Paragraph(f"Weight: {weight} kg", styles['Normal']))
    content.append(Paragraph(f"Height: {height} cm", styles['Normal']))
    content.append(Paragraph(f"Sleep: {sleep} hours", styles['Normal']))
    content.append(Paragraph(f"Sugar Intake: {sugar}", styles['Normal']))
    content.append(Paragraph(f"BMI: {bmi:.2f}", styles['Normal']))

    result = "High Risk" if prediction == 1 else "Low Risk"

    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Prediction: {result}", styles['Heading2']))
    content.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles['Normal']))

    content.append(Spacer(1, 10))
    content.append(Paragraph("Summary", styles['Heading3']))
    for a in advice:
        content.append(Paragraph(a, styles['Normal']))

    content.append(Paragraph("Diet Recommendations", styles['Heading3']))
    for d in diet:
        content.append(Paragraph(d, styles['Normal']))

    content.append(Paragraph("Lifestyle Improvements", styles['Heading3']))
    for l in lifestyle:
        content.append(Paragraph(l, styles['Normal']))

    content.append(Paragraph("Natural Remedies", styles['Heading3']))
    for r in remedies:
        content.append(Paragraph(r, styles['Normal']))

    doc.build(content)
    buffer.seek(0)
    return buffer

# ================= LOAD MODEL ================= #
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    st.error("Model or Scaler file not found.")
    st.stop()

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# ================= UI CONFIG ================= #
st.set_page_config(page_title="Smart Health AI", page_icon="🏥", layout="wide")

st.markdown("<h1 style='text-align:center; color:#2E86C1;'>🏥 Smart Health Risk Dashboard</h1>", unsafe_allow_html=True)

# ================= SIDEBAR ================= #
st.sidebar.title("🧾 Patient Input")

age = st.sidebar.slider("Age", 18, 100, 30)
weight = st.sidebar.slider("Weight (kg)", 40, 150, 70)
height = st.sidebar.slider("Height (cm)", 140, 210, 170)
sleep = st.sidebar.slider("Sleep Hours", 3, 10, 7)

exercise = st.sidebar.selectbox("Exercise?", ["No", "Yes"])
exercise = 1 if exercise == "Yes" else 0

sugar = st.sidebar.slider("Sugar Intake", 0, 10, 5)

smoking = st.sidebar.selectbox("Smoking?", ["No", "Yes"])
smoking = 1 if smoking == "Yes" else 0

alcohol = st.sidebar.selectbox("Alcohol?", ["No", "Yes"])
alcohol = 1 if alcohol == "Yes" else 0

married = st.sidebar.selectbox("Married?", ["No", "Yes"])
married = 1 if married == "Yes" else 0

profession_list = [
    "Student","Software Engineer","Doctor","Teacher","Business",
    "Engineer","Government Job","Self-Employed","Unemployed","Other"
]
profession = profession_list.index(st.sidebar.selectbox("Profession", profession_list))

# BMI
bmi = weight / ((height/100)**2)

# ================= PREDICTION ================= #
if st.button("🔍 Analyze Health Risk"):

    input_data = pd.DataFrame([{
        "age": age,
        "weight": weight,
        "height": height,
        "exercise": exercise,
        "sleep": sleep,
        "sugar_intake": sugar,
        "smoking": smoking,
        "alcohol": alcohol,
        "married": married,
        "profession": profession,
        "bmi": bmi
    }])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)
    confidence = float(np.max(prob))

    # ================= RESULT CARD ================= #
    st.markdown(f"""
    <div style="background-color:#2E86C1;padding:20px;border-radius:10px;color:white;text-align:center">
    <h2>Prediction: {"HIGH RISK" if prediction[0]==1 else "LOW RISK"}</h2>
    <h4>Confidence: {confidence*100:.2f}%</h4>
    </div>
    """, unsafe_allow_html=True)

    # ================= GRAPH ================= #
    fig, ax = plt.subplots()
    ax.bar(["Confidence"], [confidence * 100])
    ax.set_ylabel("Percentage")
    st.pyplot(fig)

    # ================= STORE IN DATABASE ================= #
    record = {
        "age": age,
        "weight": weight,
        "height": height,
        "sleep": sleep,
        "exercise": exercise,
        "sugar": sugar,
        "smoking": smoking,
        "alcohol": alcohol,
        "married": married,
        "profession": profession,
        "bmi": float(bmi),
        "prediction": int(prediction[0]),
        "confidence": confidence,
        "timestamp": datetime.now()
    }

    collection.insert_one(record)

    # ================= AI SUGGESTIONS ================= #
    st.subheader("🩺 Health Suggestions")

    advice, diet, lifestyle, remedies = get_health_advice(
        prediction[0], age, bmi, smoking, alcohol, sleep, exercise, sugar
    )

    st.write("### 📌 Summary")
    for a in advice: st.write(a)

    st.write("### 🥗 Diet")
    for d in diet: st.write(d)

    st.write("### 🏃 Lifestyle")
    for l in lifestyle: st.write(l)

    st.write("### 🌿 Remedies")
    for r in remedies: st.write(r)

    # ================= PDF ================= #
    pdf = generate_pdf(
        age, weight, height, sleep, sugar, bmi,
        prediction[0], confidence,
        advice, diet, lifestyle, remedies
    )

    st.download_button("📥 Download Report", pdf, "health_report.pdf", "application/pdf")

# ================= SHOW DATABASE RECORDS ================= #
st.markdown("## 📂 Previous Records")

data = list(collection.find().sort("timestamp", -1).limit(5))

if data:
    df = pd.DataFrame(data)
    st.dataframe(df)
else:
    st.write("No records found.")
