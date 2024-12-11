# Predicting Metastatic Triple Negative Breast Cancer (TNBC) Based on Patient Demographics
- This repository houses an attempt to apply three types of supervised machine learning models to the "WiDS Datathon 2024 Challenge #1" Kaggle challenge; Metastatic TNBC dataset.

## Overview
The challenge is to predict if a patient is diagnosed with Metastatic TNBC within a 90 day diagnostic period based on their demographics. The dataset is composed of patient information from medical records, insurance records, and environmental health I will aproach this using toxicology data from NASA/Columbia University based on patient zip-codes. To determine the correlation, between demographic and diagnostic period, classification models (Random Forest, Decision Tree, and Gradient Boosting) were used while accuracy, precision, recall, and F-1 scores were calculated to assess the model's performance.


## Summary of Work Done
- Data Overview
	- Data Type: Tabular
	- train.csv
		- 12906 rows, 83 columns
	- test.csv
		- 5792 rows, 82 columns
		- Omits the target column ('DiagPeriodL90D')
  - The target column 'DiagPeriodL90D' is the indicator for Metastatic TNBC in this dataset with 1 meaning the patient received a metastatic cancer diagnosis within 90 days of screening and 0 meaning the patient did not receive a diagnosis.
    
 ![data_visual](https://github.com/user-attachments/assets/61279ef4-def3-454b-a61c-224f04739b9c)
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
  
## Training
- Train: 1111 rows (90% of train.csv)
- Validation: 124 rows (10% of train.csv)
- Test: 824 rows (100% of test.csv)
- Random Forest (Hyperparameter Tuning via GridSearchCV)
	- n_estimators=100
	- max_depth=5,
	- random_state=42
- Decision Tree (Hyperparameter Tuning via GridSearchCV)  
	- criterion= entropy
	- max_depth=5
	- min_samples_leaf= 1
	- min_samples_split= 5

## Results

## Future Work
Moving forward, I plan on incorporating other models such as the Naive Bayes model, decision tree, and random forest. I was not successful with these models during this tiral period, but with the linear regression model as a benchmark, 


# How to Reproduce Results
1. Download the train.csv and test.csv from the kaggle challenge: (https://www.kaggle.com/competitions/widsdatathon2024-challenge1/overview)
2. Load the CSV files and run the Data Understanding notebook followed by the Preprocessing notebook in order to obtain the train_preprocessed.csv and test_prepocessed.csv.
3. Load the preprocessed CSV files and run the ML_Testing notebook to train the model and review the results.

4. If you have any feedback or suggestions for improvement, please feel free to contact me at kja7375@mavs.uta.edu
