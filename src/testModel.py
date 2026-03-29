import pandas as pd
import joblib

# Load saved model
model = joblib.load("models/best_model.pkl")

print("Model loaded successfully.")
print("Model type:", type(model))

# Create one new patient record
new_patient = pd.DataFrame([{
    "age": 65,
    "anaemia": 1,
    "creatinine_phosphokinase": 250,
    "diabetes": 0,
    "ejection_fraction": 30,
    "high_blood_pressure": 1,
    "platelets": 250000,
    "serum_creatinine": 1.8,
    "serum_sodium": 135,
    "sex": 1,
    "smoking": 0,
    "time": 120,
    "age_group": "61-70",
    "creatinine_to_sodium_ratio": 1.8 / 135
}])

# Predict class
prediction = model.predict(new_patient)

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("Predicted outcome: Higher mortality risk")
else:
    print("Predicted outcome: Lower mortality risk")

# Optional: prediction probability, if supported
if hasattr(model, "predict_proba"):
    probability = model.predict_proba(new_patient)
    print("Prediction probabilities:", probability)