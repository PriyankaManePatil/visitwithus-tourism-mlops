import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO = "priyankamane10/visitwithus-tourism-model"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="best_model.pkl",
        repo_type="model"
    )
    return joblib.load(model_path)

model = load_model()

st.title("Wellness Tourism Package Purchase Prediction")

# Input UI
st.write("Provide customer details to predict if they will purchase the wellness tourism package.")

def user_input():
    data = {
        "Age": st.number_input("Age", min_value=18, max_value=100),
        "TypeofContact": st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"]),
        "CityTier": st.selectbox("City Tier", [1, 2, 3]),
        "DurationOfPitch": st.number_input("Duration of Pitch"),
        "Occupation": st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business", "Other"]),
        "Gender": st.selectbox("Gender", ["Male", "Female"]),
        "NumberOfPersonVisiting": st.number_input("Number of People Visiting", min_value=1),
        "NumberOfFollowups": st.number_input("Followup Count"),
        "ProductPitched": st.selectbox("Product Pitched", ["Basic", "Deluxe", "Super Deluxe", "King", "Queen"]),
        "PreferredPropertyStar": st.selectbox("Preferred Property Rating", [3, 4, 5]),
        "MaritalStatus": st.selectbox("Marital Status", ["Married", "Single", "Divorced"]),
        "NumberOfTrips": st.number_input("Travel Trips/Yr"),
        "Passport": st.selectbox("Passport", [0, 1]),
        "PitchSatisfactionScore": st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5]),
        "OwnCar": st.selectbox("Owns a Car?", [0, 1]),
        "NumberOfChildrenVisiting": st.number_input("Children Traveling", min_value=0),
        "Designation": st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP", "Entry Level"]),
        "MonthlyIncome": st.number_input("Monthly Income", min_value=5000),
    }
    return pd.DataFrame([data])

input_df = user_input()

if st.button("Predict"):
    prediction_prob = model.predict_proba(input_df)[0][1]
    result = "Will Purchase" if prediction_prob > 0.5 else "Will Not Purchase"
    st.subheader(f"Prediction: {result}")
    st.write(f"Probability of Purchase: {prediction_prob:.2%}")
