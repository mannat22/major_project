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

# ================= FEATURE ORDER ================= #
FEATURES = [
    "age","weight","height","exercise","sleep",
    "sugar_intake","smoking","alcohol","married","profession","bmi"
]

# ================= AI CORE ================= #
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
        tips.append("🥗 Reduce oily & processed food")
    if bmi < 18:
        tips.append("🍗 Increase protein intake")
    if sleep < 6:
        tips.append("😴 Sleep 7–8 hours daily")
    if sugar > 7:
        tips.append("🚫 Avoid sugar & soft drinks")
    if smoking:
        tips.append("🚭 Reduce smoking gradually")
    if alcohol:
        tips.append("🍺 Limit alcohol")

    if not tips:
        tips.append("✅ Maintain healthy lifestyle")

    return tips


def health_improvement_plan(bmi, sleep, sugar, smoking, alcohol):

    plan = []

    # Diet & weight
    if bmi > 25:
        plan.append("🥗 Eat more vegetables, fiber, fruits")
        plan.append("🚶 Start daily walking 30 minutes")
    elif bmi < 18:
        plan.append("🍗 Increase protein-rich foods like eggs, milk, nuts")

    # Sleep
    if sleep < 6:
        plan.append("😴 Fix sleep schedule (7–8 hours daily)")

    # Sugar
    if sugar > 7:
        plan.append("🚫 Reduce sweets, cold drinks, packaged foods")

    # Habits
    if smoking:
        plan.append("🚭 Reduce smoking gradually (step-by-step)")
    if alcohol:
        plan.append("🍺 Reduce alcohol consumption")

    # Natural remedies (VERY IMPORTANT)
    plan.append("🌿 Drink warm lemon water every morning")
    plan.append("🌿 Drink turmeric milk before sleep")
    plan.append("🌿 Drink 2–3L water daily")
    plan.append("🌿 Do light exercise or yoga daily")
    plan.append("🌿 Green tea helps metabolism")

    return plan


# ================= PDF ================= #
def generate_pdf(name, score, level, confidence, recs, plan):

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

    for r in recs:
        content.append(Paragraph("• " + r, styles["Normal"]))

    content.append(Spacer(1, 10))
    content.append(Paragraph("Health Improvement Plan:", styles["Heading2"]))

    for p in plan:
        content.append(Paragraph("• " + p, styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer


# ================= MAIN ================= #
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Patient Overview")
st.write(f"👤 Name: {name}")
st.write(f"Age: {age} | BMI: {bmi:.2f}")
st.markdown('</div>', unsafe_allow_html=True)

# ================= ANALYZE ================= #
if st.button("🔍 Analyze Health Risk"):

    if name.strip() == "":
        st.warning("Please enter name!")
        st.stop()

    # SAFE INPUT (NO ERROR)
    input_array = np.array([[
        age, weight, height, exercise, sleep,
        sugar, smoking, alcohol, married, profession, bmi
    ]])

    input_scaled = scaler.transform(input_array)

    prediction = int(model.predict(input_scaled)[0])
    confidence = float(np.max(model.predict_proba(input_scaled)))

    score, level = health_score(prediction, bmi, sleep, sugar, smoking, alcohol)

    recs = ai_recommendations(bmi, sleep, sugar, smoking, alcohol)
    plan = health_improvement_plan(bmi, sleep, sugar, smoking, alcohol)

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

    # ================= PLAN ================= #
    st.markdown("### 🌿 Health Improvement Plan")
    for p in plan:
        st.write("🌿", p)

    # ================= PDF ================= #
    pdf = generate_pdf(name, score, level, confidence, recs, plan)

    st.download_button(
        "📥 Download Health Report",
        pdf,
        file_name="health_report.pdf",
        mime="application/pdf"
    )
