
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("health_risk_model.pkl")
scaler = joblib.load("scaler.pkl")

conn = sqlite3.connect("health_predictions.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions(
id INTEGER PRIMARY KEY AUTOINCREMENT,
age INTEGER,
bmi REAL,
sleep INTEGER,
exercise INTEGER,
prediction INTEGER,
risk_score INTEGER,
uncertainty REAL
)
""")

conn.commit()

st.title("Smart Health Risk Prediction System")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Health Prediction", "Health Dashboard", "Population Analytics"]
)

if page == "Health Prediction":

    st.header("Enter Health Information")

    age = st.number_input("Age", 0, 100)
    weight = st.number_input("Weight (kg)")
    height = st.number_input("Height (meters)")
    sleep = st.number_input("Sleep Hours", 0, 24)
    exercise = st.number_input("Exercise per week", 0, 7)
    sugar = st.number_input("Sugar Intake (0-10)", 0, 10)

    smoking = st.selectbox("Smoking", ["no", "yes"])
    alcohol = st.selectbox("Alcohol", ["no", "yes"])
    married = st.selectbox("Married", ["no", "yes"])

    bmi = weight / (height ** 2) if height > 0 else 0

    if st.button("Predict Health Risk"):

        input_data = pd.DataFrame([[age, weight, height, exercise, sleep, sugar,
        1 if smoking=="yes" else 0,
        1 if alcohol=="yes" else 0,
        1 if married=="yes" else 0,
        bmi]],
        columns=["age","weight","height","exercise","sleep","sugar_intake",
        "smoking","alcohol","married","bmi"])

        input_scaled = scaler.transform(input_data)

        prob = model.predict_proba(input_scaled)
        prediction = model.predict(input_scaled)[0]

        confidence = np.max(prob)
        uncertainty = 1 - confidence
        risk_score = int(confidence * 100)

        st.subheader("Prediction Result")

        if prediction == 0:
            st.success("Low Health Risk")
        elif prediction == 1:
            st.warning("Medium Health Risk")
        else:
            st.error("High Health Risk")

        st.write("Risk Score:", risk_score)
        st.write("Confidence:", round(confidence,2))
        st.write("Uncertainty:", round(uncertainty,2))

        cursor.execute("""
        INSERT INTO predictions(age,bmi,sleep,exercise,prediction,risk_score,uncertainty)
        VALUES(?,?,?,?,?,?,?)
        """,(age,bmi,sleep,exercise,int(prediction),risk_score,uncertainty))

        conn.commit()
