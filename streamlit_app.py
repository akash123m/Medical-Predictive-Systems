import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
url = "https://drive.google.com/file/d/14pqzFDCs5c93arLZJ2Q8msrzIOZjldK4/view?usp=sharing"
url='https://drive.google.com/uc?id=' + url.split('/')[-2]
diabetes_dataset = pd.read_csv(url)
X = diabetes_dataset.drop(columns = 'Outcome', axis =1)
Y = diabetes_dataset['Outcome']
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel = 'linear')
classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("Training Accuracy Score: ", training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print("Testing Accuracy Score: ", test_data_accuracy)

@st.cache()
def prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    input_data_np = np.asarray(input_data)
    input_data_reshaped = input_data_np.reshape(1,-1)
    std_data = scaler.transform(input_data_reshaped)
    result = classifier.predict(std_data)
    if result[0] == 0:
        return "Non-Diabetic"
    else:
        return "Diabetic"

st.title("Diabetes Prediction App")

Pregnancies = st.slider("Pregnancies", 0.0, 20.0)
Glucose = st.slider("Glucose", 0.0, 250.0)
BloodPressure = st.slider("Blood Pressure", 0.0, 130.0)
SkinThickness = st.slider("Skin Thickness", 0.0, 101.0)
Insulin = st.slider("Insulin", 0.0, 900.0)
BMI = st.slider("BMI", 0.0, 70.0)
DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", 0.0, 3.0)
Age = st.slider("Age", 16.0, 90.0)

if st.button("Predict"):
	diagnosis = prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
	st.write("Diagnosis predicted:", diagnosis)
	st.write("Accuracy score of this model is:", test_data_accuracy)
