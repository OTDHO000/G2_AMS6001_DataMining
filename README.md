# Heart Attack Analysis & Prediction (R)

## A dataset for heart attack classification in kaggle
https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/code?datasetId=1226038&searchQuery=clust#

In this analysis ,I want to embark on a comprehensive exploratory data analysis (EDA) journey, discerning the distribution of individual features, and examining the relationships among them. 
I aim to employ cluster analysis, a method that allows us to group similar data instances based on their feature similarities. This could reveal underlying patterns or subgroups within the data, which might be indicative of specific risk groups or heart disease profiles.

The dataset, composed of both categorical and numerical data, provides a comprehensive snapshot of factors potentially contributing to heart diseases, including demographic details like ageand sex, clinical parameters like chest pain type (cp), resting blood pressure (trtbps), serum cholesterol (chol), fasting blood sugar (fbs), and resting electrocardiographic results (restecg), among others. Each feature encapsulates a different aspect of an individual's health profile, potentially playing a role in heart disease development. Moreover, the inclusion of the output variable indicating the diagnosis of heart disease presents an opportunity to develop predictive models that could help in early detection and prevention of this deadly disease.
## About this dataset
1. Age : Age of the patient
2. Sex : Sex of the patient
3. exang: exercise induced angina (1 = yes; 0 = no)
4. caa: number of major vessels (0-3)
5. cp : Chest Pain type chest pain type
  Value 1: typical angina
  Value 2: atypical angina
  Value 3: non-anginal pain
  Value 4: asymptomatic
6. trtbps : resting blood pressure (in mm Hg)
7. chol : cholestoral in mg/dl fetched via BMI sensor
8. fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
9. rest_ecg : resting electrocardiographic results
  Value 0: normal
  Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
  Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
10. thalach : maximum heart rate achieved
11. ST Depression Induced by Exercise Relative to Rest (oldpeak) : Changes in the ST segment on an ECG can indicate heart disease. ST depression can indicate ischemia, or lack of sufficient blood flow to the heart muscle.
12. The Slope of The Peak Exercise ST Segment (slp) : The ST segment/heart rate slope (ST/HR slope), has been introduced as an index of relative myocardial oxygen demand during exercise. The shape of the ST segment can reveal a lot about the heart's condition.
13. Thallium Stress Test (thall) : This is a nuclear imaging method that shows how well blood flows into the heart muscle, both at rest and during activity. It can reveal areas of the heart muscle that aren't receiving enough blood, indicating coronary artery disease.
target : 0= less chance of heart attack 1= more chance of heart attack

