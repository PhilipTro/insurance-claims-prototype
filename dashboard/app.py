import streamlit as st
import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path.home() / "Desktop" / "Insurance_Project"

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
with st.form("prediction_form"):
    months_as_customer = st.number_input("Months as Customer", min_value=0, max_value=600, value=12)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    policy_state = st.selectbox("Policy State", ['OH', 'NY', 'CA', 'FL', 'TX'])  # example states
    policy_csl = st.selectbox("Policy CSL", ['100/300', '250/500', '500/1000'])
    policy_deductable = st.number_input("Policy Deductable", min_value=0, max_value=5000, value=500)
    policy_annual_premium = st.number_input("Policy Annual Premium", min_value=0, max_value=10000, value=1200)
    umbrella_limit = st.number_input("Umbrella Limit", min_value=0, max_value=1000000, value=0)
    insured_sex = st.selectbox("Insured Sex", ['Male', 'Female'])
    insured_education_level = st.selectbox("Education Level", ['High School', 'College', 'Masters', 'PhD'])
    insured_occupation = st.text_input("Occupation", value="Engineer")
    insured_relationship = st.selectbox("Relationship", ['Single', 'Married', 'Other'])
    incident_state = st.selectbox("Incident State", ['OH', 'NY', 'CA', 'FL', 'TX'])
    witnesses = st.number_input("Number of Witnesses", min_value=0, max_value=10, value=0)
    auto_make = st.text_input("Auto Make", value="Toyota")
    auto_model = st.text_input("Auto Model", value="Camry")
    auto_year = st.number_input("Auto Year", min_value=1980, max_value=2025, value=2020)
    month_sin = st.slider("Month Sin", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
    month_cos = st.slider("Month Cos", min_value=-1.0, max_value=1.0, value=1.0, step=0.01)
    
    submit = st.form_submit_button("Predict")
def predict_model(X):
    pass
if submit:
    # Collect inputs into a dataframe
    input_data = pd.DataFrame([{
        'months_as_customer': months_as_customer,
        'age': age,
        'policy_state': policy_state,
        'policy_csl': policy_csl,
        'policy_deductable': policy_deductable,
        'policy_annual_premium': policy_annual_premium,
        'umbrella_limit': umbrella_limit,
        'insured_sex': insured_sex,
        'insured_education_level': insured_education_level,
        'insured_occupation': insured_occupation,
        'insured_relationship': insured_relationship,
        'incident_state': incident_state,
        'witnesses': witnesses,
        'auto_make': auto_make,
        'auto_model': auto_model,
        'auto_year': auto_year,
        'month_sin': month_sin,
        'month_cos': month_cos
    }])

    model_features = ['months_as_customer', 'age', 'policy_state', 'policy_csl',
       'policy_deductable', 'policy_annual_premium', 'umbrella_limit',
       'insured_sex', 'insured_education_level', 'insured_occupation',
       'insured_relationship', 'incident_state', 'witnesses', 'auto_make',
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
