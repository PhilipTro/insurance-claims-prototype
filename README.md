# Insurance_Project
A project meant to simulate real-world insurance risk segmentation and claim prediction

## Approach

The restricitons surrouding insurance data has necessitated prototyping with limited data. 
The purpose of the project is to calculate a customers insurance premium by predicting insurance claim likelihood and insurance claim severity.

## Structure

The project is structured into the following sections: EDA, data preprocessing, clustering, regression and classification. 

The EDA and data preprocessing sections are of vital importance in ensuring the validity and relevance of the analysed data, while the purpose of the clustering section is to explore if there are any possible segmentations of insurance policyholders.

The purpose of the classification section is to predict to probability of a claim being filed, while the purpose of the regression section is to predict the severity of said claim. These two predicted values will then be used in order to calculate an insurance premium. 

The regression section utilizes TweddieRegressor, and the classification section utilises ensemble models such as XGBClassifier and RandomForest. 

The project also inlcudes a dashboard built using streamlit on the prototype data, with the aim of upscaling to a dashboard built using Tableau.

## Current Stage
The prototype, built on a limited amount of data, is currently implemented. The next stage is identifying a suitable and more extensive dataset and reworking the prototype with considerations taken for the new dataset.
