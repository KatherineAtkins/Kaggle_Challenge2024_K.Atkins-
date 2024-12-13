# Predicting Metastatic Triple Negative Breast Cancer (TNBC) Based on Patient Demographics
- This repository houses an attempt to apply three types of supervised machine learning models to the "WiDS Datathon 2024 Challenge #1" Kaggle challenge; Metastatic TNBC dataset.

## Overview
The challenge is to predict if a patient is diagnosed with Metastatic TNBC within a 90 day diagnostic period based on their demographics. The dataset is composed of patient information from medical records, insurance records, and environmental health I will aproach this using toxicology data from NASA/Columbia University based on patient zip-codes. To determine the correlation, between demographic and diagnostic period, classification models (Random Forest, Decision Tree, and Logistic Regression) were used while accuracy, precision, recall, and F-1 scores were calculated to assess the model's performance. The highest predicted accuracy score from these models was 0.601 via Random Forest Model while the highest accuracy score from the Kaggle Challenge is 0.802 via CatBoost and XGBoost. More information about the performance of these models can be found below.


## Summary of Work Done
- Data Overview
	- Data Type: Tabular
	- train.csv
		- 12906 rows, 83 columns
	- test.csv
		- 5792 rows, 82 columns
		- Omits the target column ('DiagPeriodL90D')
  - The target column 'DiagPeriodL90D' is the indicator for Metastatic TNBC in this dataset with 1 meaning the patient received a metastatic cancer diagnosis within 90 days of screening and 0 meaning the patient did not receive a diagnosis.
    
![stack_visual](https://github.com/user-attachments/assets/aa8d4388-5202-4cb6-b5b8-3b710204657d)
![diagnosisbystate](https://github.com/user-attachments/assets/294499c8-cfcb-4aea-a27c-04fe8b15fd3f)
![ozonebystate](https://github.com/user-attachments/assets/921c7340-d64e-457d-8dc2-4deee2c6f9da)
![PM25bystate](https://github.com/user-attachments/assets/68e47020-eda1-4a71-b08e-24b6239b927b)
![NO2bystate](https://github.com/user-attachments/assets/4c09c7ec-0f79-4a01-a213-b6d44923dec1)

  
## Preprocessing
- Duplicates: N/A
- Outliers: N/A
- Drop Outright Unnecessary Columns: ("education_less_highschool","education_highschool","education_some_college","education_bachelors","education_graduate","education_college_or_above","education_stem_degree","married","divorced","never_married","widowed","age_median")
- Adjust age_range: '19 and Under', '20-39', '40-59', '60-69', 'Over 70'
- Replace missing values in 'patient_race' column with 'Prefer Not to Answer' category
- Replace NaN 'BMI' values with the average BMI based on 'age_group', 'patient_race', 'patient_state' via groupby
- Feature Engineer
	- Create a new feature based on the locations cancers ('breast_cancer_diagnosis_desc')
- Encode Categorical Features (One-Hot Encoding)
	- 'payer_type','patient_state', 'location_category'
- Additional drop of unnecessary columns: ('mean_bmi','breast_cancer_diagnosis_desc','age_group','age_under_10','age_10_to_19','age_20s','age_30s','age_30s','age_40s','age_50s','age_60s','age_70s','age_over_80','patient_age','patient_race','patient_gender','breast_cancer_diagnosis_code','breast_cancer_diagnosis_code', 'metastatic_cancer_diagnosis_code','metastatic_first_novel_treatment','metastatic_first_novel_treatment_type','Region','Division','labor_force_participation','unemployment_rate','self_employed','farmer','payer_type_MEDICAID','payer_type_MEDICARE ADVANTAGE','payer_type_Prefer Not to Answer','family_size')
	- #### The dataset had already been encoded for quite a few features, but I didn't like the grouping of some of them, so I redid them and deleted the original ones. Additional elimination was done after doing a correlation matrix assessment and assessing the necessity of most features by what I deemed more relevant to the demographic of the patients.
  
## Training
- Train: 90% of train.csv
- Validation: 10% of train.csv
- Test: 100% of test.csv
	- Output is predictions
- Random Forest
	- n_estimators=100
	- max_depth=5,
	- random_state=42
- Decision Tree 
	- criterion= entropy
	- max_depth=5
	- random_state=42
 - Decision Tree
	- random_state=42

-  #### Training was the most difficult (and discouraging) aspect of this project as models would work well on one run, but not another after resetting the kernels, resulting in the loss of progress. My models seemed resistent to manual hypertuning as they would not change results when significantly tuned.

## Results and Conclusion
![python2table](https://github.com/user-attachments/assets/c1641230-e9bf-4615-8a97-890519f79eab)

The results are very mid-field and have similar ROC curves to random (50/50) selection as shown below.

![ROCrandomforest](https://github.com/user-attachments/assets/52a6d95d-bef8-4350-a5f6-a8107375c8c1)
![ROCdecisiontree](https://github.com/user-attachments/assets/d2410bdb-a929-40e1-af22-fb493c18002b)
![ROClogreg](https://github.com/user-attachments/assets/2467dcb7-a372-4822-904f-b4edf79b92c6)

The best score for this Kaggle Challenge is an accuracy of 0.802, making my scores pale in comparison. The models shown in this project were insignificant in correctly identifying patients who are likely to develop Metastatic TNBC based on demographics. This is likely due to my choice in handling the data during preprocessing and feature engineering. 

## Future Work
Moving forward, I plan to reassess the preprocessing stage as I aim to incorporate other models such as the Naive Bayes model and Gradient Boosting in addition to solidifying benchmark results most likely with a linear regression model. I unfortunately was not successful with these models during this tiral period... :(  Additionally, I plan to tune the modles with hyperparameters via GridSearchCV and cross validate the Random Forest, Decision Tree, and Logistic Regression models already used in this project.


# How to Reproduce Results
1. Download the train.csv and test.csv from the Kaggle Challenge "WiDS Datathon 2024 Challenge #1": (https://www.kaggle.com/competitions/widsdatathon2024-challenge1/overview)
	- Train and test datasets need to be downloaded seperately
2. Load the CSV files and run the Data Understanding notebook followed by the Preprocessing notebook in order to obtain the train_preprocessed.csv and test_prepocessed.csv.
3. Load the preprocessed CSV files and run the ML_Testing notebook to train the model and review the results.


## Overview of files in repository
- DataUnderstanding.ipynb: This notebook will walk you through how to initially look at the data including statistics and visualizations.
- Preprocessing-2.ipynb: This notebook focuses on preprocessing both the training and test datasets with a primary focus on the training data exploration. This notebook includes feature selection, encoding, and feature engineering.
- TrainingML_Visualizations: This notebook focuses only on the training dataset and visualizes the the Random Forest and Decicion Tree models. *You do NOT need to follow this notebook to reproduce results, it is purley for entertainment only*
- ML_Testing.ipynb: This notebook contains the model training and testing of Random Forest, Decision Tree, and Logistic Regression as well as ROC curves for each model.

Software Setup
- I accessed all software through libraries

- Data Understanding and Preprocessing
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
	import re
- Machine Learning
	from sklearn.model_selection import train_test_split
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	from sklearn.model_selection import GridSearchCV
	from sklearn.metrics import roc_curve, auc
	from sklearn.metrics import mean_squared_error, r2_score
	from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
	from sklearn.tree import DecisionTreeClassifier, plot_tree
	from sklearn.compose import ColumnTransformer
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import OneHotEncoder, StandardScaler
	from sklearn.metrics import classification_report




#### If you have any feedback or suggestions for improvement, please feel free to contact me at kja7375@mavs.uta.edu
