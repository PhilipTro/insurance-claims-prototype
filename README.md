# Insurance Project
A prototype project simulating real-world insurance risk segmentation and claim prediction.

## Overview
This project aims to calculate insurance premiums by predicting:
1. The likelihood of a customer filing a claim (classification)  
2. The severity of the claim (regression)

Due to restrictions on real insurance data, the project is currently prototyped with a limited dataset.

## Approach
The project workflow includes:
- **Exploratory Data Analysis (EDA) & Data Preprocessing:** Ensures data validity and relevance.  
- **Clustering:** Explores potential segmentations of policyholders.  
- **Classification:** Predicts the probability of a claim being filed using ensemble models (XGBClassifier, RandomForest).  
- **Regression:** Predicts claim severity using TweddieRegressor.  
- **Premium Calculation:** Combines predicted probability and severity to estimate insurance premiums.  
- **Dashboard:** Prototype dashboard implemented with Streamlit, with plans to scale to Tableau.

## Skills & Technologies
Python, pandas, scikit-learn, XGBoost, Streamlit, data preprocessing, EDA, regression, classification, clustering.

## Current Stage
The prototype is implemented with a limited dataset. The next step is to identify a larger, more suitable dataset and rework the prototype accordingly.

## Project Highlights
- Built a working prototype for insurance claim prediction and premium calculation.
- Implemented classification (XGBClassifier, RandomForest) and regression (TweddieRegressor) pipelines.
- Developed an interactive Streamlit dashboard to explore predictions and visualize statistics.


## Screenshot of dashboard with quantitative statistics
<img alt="image" src="https://github.com/user-attachments/assets/9af0f279-a61d-497c-90b6-5888e3a9b910" />

## Screenshot of mock prediction menu in dashboard
<img alt="image" src="https://github.com/user-attachments/assets/36a5f124-8a47-4473-900c-14524f094d37" />


