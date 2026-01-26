import streamlit as st
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path.home() / "Desktop" / "Insurance_Project" / "prototype"

# Add root to Python path
sys.path.append(str(PROJECT_ROOT))

st.title("Insurance Claims Dashboard")
df = pd.read_csv(PROJECT_ROOT/"dashboard"/"assets"/"dashboard_data.csv")

# Show raw data
if st.checkbox("Show raw data"):
    st.dataframe(df.head())

# Sidebar filters
policy_state = st.sidebar.selectbox("Select State", df["policy_state"].unique())
filtered_df = df[df['policy_state'] == policy_state]

#KPI's

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total policies", len(filtered_df))
with col2:
    st.metric("Total claim amount", int(filtered_df['total_claim_amount'].sum()))
with col3:
    st.metric("Average Claim amount", round(filtered_df['total_claim_amount'].mean(), 2))



st.subheader("Correlation Heatmap")
num_cols = df.select_dtypes('number').columns
corr = filtered_df[num_cols].corr()
fig, ax = plt.subplots()
sns.heatmap(corr, cmap='RdYlBu_r', ax=ax)
st.pyplot(fig)


st.title("Insurance Claim Prediction")

preprocessor = joblib.load(PROJECT_ROOT/"models"/"preprocessor.pkl")
model = joblib.load(PROJECT_ROOT/"models"/"regression"/"tweedie_model.pkl")


# Create input form
policy_state_cats = ['OH', 'IN', 'IL']
policy_csl_cats = ['100/300', '250/500', '500/1000']
insured_sex_cats = ['MALE', 'FEMALE']
education_cats = ['High School', 'Masters', 'JD', 'MD']

with st.form("prediction_form"):

    # --- Customer Info ---
    st.subheader("Customer Information")
    months_as_customer = st.number_input("Months as Customer", 0, 600, 12)
    age = st.number_input("Age", 18, 100, 30)
    insured_zip = st.text_input("Insured ZIP Code", value="12345")

    # --- Policy Info ---
    st.subheader("Policy Information")
    policy_state = st.selectbox("Policy State", policy_state_cats)
    policy_csl = st.selectbox("Policy CSL", policy_csl_cats)
    policy_deductable = st.number_input("Policy Deductible", 0, 5000, 500)
    policy_annual_premium = st.number_input("Annual Premium", 0, 20000, 1200)
    umbrella_limit = st.number_input("Umbrella Limit", 0, 1000000, 0)

    # --- Insured Info ---
    st.subheader("Insured Profile")
    insured_sex = st.selectbox("Sex", insured_sex_cats)
    insured_education_level = st.selectbox("Education Level", education_cats)
    insured_occupation = st.text_input("Occupation", value="Engineer")
    insured_relationship = st.selectbox("Relationship Status",
                                        ['Single', 'Married', 'Other'])

    # --- Incident Info ---
    st.subheader("Incident Information")
    incident_state = st.selectbox("Incident State", ['OH', 'IN', 'IL'])
    incident_city = st.text_input("Incident City", value="Columbus")
    incident_location = st.text_input("Incident Location", value="123 Main St")
    witnesses = st.number_input("Number of Witnesses", 0, 10, 0)

    # --- Vehicle Info ---
    st.subheader("Vehicle Information")
    auto_make = st.text_input("Auto Make", value="Toyota")
    auto_model = st.text_input("Auto Model", value="Camry")
    auto_year = st.number_input("Auto Year", 1980, 2030, 2020)

    # --- Cyclic Encodings ---
    st.subheader("Cyclic Month Encoding")
    month_sin = st.slider("Month Sin", -1.0, 1.0, 0.0, 0.01)
    month_cos = st.slider("Month Cos", -1.0, 1.0, 1.0, 0.01)

    submit = st.form_submit_button("Predict")

def predict_model(X):
    pass
if submit:
    input_data = pd.DataFrame([{
        'months_as_customer': months_as_customer,
        'age': age,
        'policy_state': policy_state,
        'policy_csl': policy_csl,
        'policy_deductable': policy_deductable,
        'policy_annual_premium': policy_annual_premium,
        'umbrella_limit': umbrella_limit,
        'insured_zip': insured_zip,
        'insured_sex': insured_sex,
        'insured_education_level': insured_education_level,
        'insured_occupation': insured_occupation,
        'insured_relationship': insured_relationship,
        'incident_state': incident_state,
        'incident_city': incident_city,
        'incident_location': incident_location,
        'witnesses': witnesses,
        'auto_make': auto_make,
        'auto_model': auto_model,
        'auto_year': auto_year,
        'month_sin': month_sin,
        'month_cos': month_cos
    }])


    model_features = ['months_as_customer', 'age', 'policy_state', 'policy_csl',
       'policy_deductable', 'policy_annual_premium', 'umbrella_limit',
       'insured_zip', 'insured_sex', 'insured_education_level',
       'insured_occupation', 'insured_relationship', 'incident_state',
       'incident_city', 'incident_location', 'witnesses', 'auto_make',
       'auto_model', 'auto_year', 'month_sin', 'month_cos']
    
    input_data = input_data[model_features]
    input_transformed = preprocessor.transform(input_data)

    prediction = model.predict(input_transformed)
    st.success(f"Prediction{prediction}")
#interactive prediction
#st.subheader("Claim Prediction Demo")
#age = st.slider("Policyholder Age", 18, 80)
#coverage = st.selectbox("Coverage Type", df['fraud_reported'].unique())

# For demo, mock prediction
#risk_score = (age / 80) * 0.5 + (1 if coverage == 0 else 0) * 0.5
#st.write(f"Predicted Claim Risk Score: {risk_score:.2f}")
