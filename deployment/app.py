import streamlit as st
print("--- APP STARTING ---", flush=True)
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os
import numpy as np
import yaml
from dotenv import load_dotenv

# Load configuration
load_dotenv()
# We need to make sure config.yaml is available in deployment or loaded differently
# For simplicity, assuming it's copied or we hardcode defaults if missing, 
# but best practice is to include it in Docker image.
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    HF_USERNAME = os.getenv("HF_USERNAME", config['project']['hf_username'])
    MODEL_REPO = f"{HF_USERNAME}/{config['model']['model_name']}"
    MODEL_FILENAME = config['model']['filename']
    COLUMNS_FILENAME = config['model']['columns_filename']
except:
    # Fallback if config not found (e.g. inside Docker if not copied correctly yet)
    HF_USERNAME = os.getenv("HF_USERNAME", "RavikanthAI9")
    MODEL_REPO = f"{HF_USERNAME}/tourism-model"
    MODEL_FILENAME = "model.pkl"
    COLUMNS_FILENAME = "model_columns.pkl"

@st.cache_resource
def load_artifacts():
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
        columns_path = hf_hub_download(repo_id=MODEL_REPO, filename=COLUMNS_FILENAME)
        
        model = joblib.load(model_path)
        columns = joblib.load(columns_path)
        return model, columns
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None

def main():
    st.set_page_config(page_title="Tourism Prediction", layout="centered")
    st.title("Tourism Package Purchase Prediction")
    st.write("Enter details to predict if a customer will purchase the package.")
    
    print("Starting load_artifacts()...", flush=True)
    model, columns = load_artifacts()
    print("Finished load_artifacts().", flush=True)
    
    if model and columns:
        with st.form("input_form"):
            st.subheader("Customer Demographics")
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", 18, 100, 30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
                city_tier = st.selectbox("City Tier", [1, 2, 3])
                
            with col2:
                monthly_income = st.number_input("Monthly Income", 0, 100000, 20000)
                occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
                passport = st.checkbox("Has Passport?")
                own_car = st.checkbox("Owns Car?")
                number_of_children = st.number_input("Number of Children Visiting", 0, 10, 0)

            st.markdown("---")
            st.subheader("Pitch Details")
            col3, col4 = st.columns(2)
            
            with col3:
                duration_of_pitch = st.number_input("Duration of Pitch (min)", 0, 120, 15)
                number_of_followups = st.number_input("Number of Followups", 0, 10, 3)
                product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
                pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
            
            with col4:
                type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
                preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
                number_of_trips = st.number_input("Number of Trips", 0, 20, 2)
                number_of_person_visiting = st.number_input("Number of Person Visiting", 1, 10, 2)
                designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "VP", "AVP"])

            submit = st.form_submit_button("Predict Probability")
            
            if submit:
                # Construct dataframe with 0s for all columns
                input_df = pd.DataFrame(index=[0], columns=columns)
                input_df = input_df.fillna(0) # Fill all with 0 initially
                
                # Digital mappings
                input_df['Age'] = age
                input_df['CityTier'] = city_tier
                input_df['DurationOfPitch'] = duration_of_pitch
                input_df['NumberOfPersonVisiting'] = number_of_person_visiting
                input_df['NumberOfFollowups'] = number_of_followups
                input_df['PreferredPropertyStar'] = preferred_property_star
                input_df['NumberOfTrips'] = number_of_trips
                input_df['Passport'] = int(passport)
                input_df['PitchSatisfactionScore'] = pitch_satisfaction_score
                input_df['OwnCar'] = int(own_car)
                input_df['NumberOfChildrenVisiting'] = number_of_children
                input_df['MonthlyIncome'] = monthly_income
                
                # One-hot mappings
                # Helper function to set one-hot
                def set_one_hot(col_prefix, value):
                    col_name = f"{col_prefix}_{value}"
                    if col_name in input_df.columns:
                        input_df[col_name] = 1
                
                set_one_hot("TypeofContact", type_of_contact)
                set_one_hot("Occupation", occupation)
                set_one_hot("Gender", gender)
                set_one_hot("ProductPitched", product_pitched)
                set_one_hot("MaritalStatus", marital_status)
                set_one_hot("Designation", designation)
                
                # Debug input
                print("Input Data Row:\n", input_df.iloc[0][input_df.iloc[0] > 0], flush=True)

                try:
                    prediction = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0][1]
                    
                    st.markdown("---")
                    if prediction == 1:
                        st.success(f"**Prediction: Will Purchase**")
                    else:
                        st.error(f"**Prediction: Will Not Purchase**")
                        
                    st.progress(float(proba))
                    st.write(f"**Probability of Purchase:** {proba:.2%}")
                    
                except Exception as e:
                    st.error(f"Prediction logic error: {e}")

if __name__ == "__main__":
    main()
