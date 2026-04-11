from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ---------------- PDF FUNCTION ---------------- #
def generate_pdf(age, weight, height, sleep, sugar, bmi, prediction, confidence):
    
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

    content.append(Spacer(1, 10))

    result = "High Risk" if prediction == 1 else "Low Risk"

    content.append(Paragraph(f"Prediction: {result}", styles['Heading2']))
    content.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles['Normal']))

    doc.build(content)

    buffer.seek(0)
    return buffer

# ---------------- LOAD MODEL SAFELY ---------------- #
if not os.path.exists("model.pkl") or not os.path.exists("scaler.pkl"):
    st.error("Model or Scaler file not found. Please check deployment.")
    st.stop()

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# ---------------- UI ---------------- #
st.set_page_config(page_title="Smart Health AI", layout="wide")

st.markdown("<h1 style='text-align:center; color:#2E86C1;'>🏥 Smart Health Risk Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### AI-based Prediction with Confidence + Explainability")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧾 Patient Details")

    age = st.slider("Age", 18, 100, 30)
    weight = st.slider("Weight (kg)", 40, 150, 70)
    height = st.slider("Height (cm)", 140, 210, 170)
    sleep = st.slider("Sleep Hours", 3, 10, 7)

    exercise = st.selectbox("Exercise Level", [0,1,2])
    sugar = st.slider("Sugar Intake", 0, 10, 5)

with col2:
    st.subheader("⚕️ Lifestyle Info")

    smoking = st.selectbox("Smoking", [0,1])
    alcohol = st.selectbox("Alcohol", [0,1])
    married = st.selectbox("Married", [0,1])
    profession = st.selectbox("Profession", list(range(10)))

# BMI
bmi = weight / ((height/100)**2)

# ---------------- PREDICTION ---------------- #
if st.button("🔍 Analyze Health Risk"):

    # ✅ FIXED: Use DataFrame with correct column names
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
    confidence = np.max(prob)

    st.markdown("## 🧠 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ HIGH HEALTH RISK")
    else:
        st.success("✅ LOW HEALTH RISK")

    st.progress(int(confidence * 100))
    st.write(f"Confidence Score: {confidence*100:.2f}%")

    if confidence < 0.6:
        st.warning("⚠️ Model is uncertain. Consider additional tests.")

    # ---------------- CHART ---------------- #
    st.markdown("## 📊 Health Indicators")

    labels = ['Age','Weight','Sleep','Sugar','BMI']
    values = [age, weight, sleep, sugar, bmi]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    st.pyplot(fig)

    # ---------------- SHAP (SAFE MODE) ---------------- #
    st.markdown("## 🧠 AI Explanation")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)

        fig2, ax2 = plt.subplots()
        shap.summary_plot(shap_values, input_scaled, show=False)
        st.pyplot(fig2)
    except:
        st.warning("SHAP explanation not available for this model.")

    # ---------------- PDF DOWNLOAD ---------------- #
    pdf = generate_pdf(age, weight, height, sleep, sugar, bmi, prediction[0], confidence)

    st.download_button(
        label="📥 Download Health Report (PDF)",
        data=pdf,
        file_name="health_report.pdf",
        mime="application/pdf"
    )
