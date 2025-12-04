import streamlit as st
import pandas as pd
import seaborn as sns
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

#interactive prediction
st.subheader("Claim Prediction Demo")
age = st.slider("Policyholder Age", 18, 80)
coverage = st.selectbox("Coverage Type", df['fraud_reported'].unique())

# For demo, mock prediction
#risk_score = (age / 80) * 0.5 + (1 if coverage == 0 else 0) * 0.5
#st.write(f"Predicted Claim Risk Score: {risk_score:.2f}")
