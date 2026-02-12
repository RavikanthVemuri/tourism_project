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
    
    model, columns = load_artifacts()
    
    if model and columns:
        with st.form("input_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", 18, 100, 30)
                income = st.number_input("Monthly Income", 0, 100000, 20000)
                pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
                
            with col2:
                duration = st.number_input("Duration of Pitch (min)", 0, 120, 15)
                followups = st.number_input("Number of Followups", 0, 10, 3)
                trips = st.number_input("Number of Trips", 0, 20, 2)

            # Categorical inputs (simplified for demo)
            # In a real app we'd map these to the one-hot columns
            st.markdown("---")
            passport = st.checkbox("Has Passport?")
            own_car = st.checkbox("Owns Car?")
            
            # Simple categorical dropdowns that we might not use directly but show intent
            city_tier = st.selectbox("City Tier", [1, 2, 3])
            
            submit = st.form_submit_button("Predict Probability")
            
            if submit:
                # Construct dataframe with 0s for all columns
                input_df = pd.DataFrame(columns=columns)
                input_df.loc[0] = 0
                
                # Fill known numericals directly (assuming names match regex logic or exact match)
                # We need to map our inputs to the columns.
                # Since we used pd.get_dummies, names are like 'Age', 'DurationOfPitch', etc.
                
                if 'Age' in columns: input_df['Age'] = age
                if 'MonthlyIncome' in columns: input_df['MonthlyIncome'] = income
                if 'PitchSatisfactionScore' in columns: input_df['PitchSatisfactionScore'] = pitch_score
                if 'DurationOfPitch' in columns: input_df['DurationOfPitch'] = duration
                if 'NumberOfFollowups' in columns: input_df['NumberOfFollowups'] = followups
                if 'NumberOfTrips' in columns: input_df['NumberOfTrips'] = trips
                if 'Passport' in columns: input_df['Passport'] = int(passport)
                if 'OwnCar' in columns: input_df['OwnCar'] = int(own_car)
                if 'CityTier' in columns: input_df['CityTier'] = city_tier
                
                # For One-Hot features (e.g. Gender_Male), we can't easily map without more UI logic.
                # We will just predict based on these main features for the MVP.
                
                try:
                    prediction = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0][1]
                    
                    st.success(f"Prediction: {'Will Purchase' if prediction == 1 else 'Will Not Purchase'}")
                    st.progress(float(proba))
                    st.write(f"Probability of Purchase: {proba:.2%}")
                    
                except Exception as e:
                    st.error(f"Prediction logic error: {e}")

if __name__ == "__main__":
    main()
