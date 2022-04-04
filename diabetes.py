import streamlit as st 
import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report

st.title('Diabetes Predictor')

st.sidebar.write("""
This is a web app demo using python libraries such as Streamlit, Sklearn etc
""")

st.sidebar.write ("For more info, please contact:")

st.sidebar.write("<a href='https://www.linkedin.com/in/ammar-rosdi-b24b52208/'>Ammar Rosdi </a>", unsafe_allow_html=True)

a = st.slider('Please insert your BMI',min_value= 15.0, max_value = 50.0, value=0.2)
b = st.slider('Please insert your Pregnancies',min_value= 1.0, max_value = 10.0, value=1.0)
c = st.slider('Please insert your Glucose',min_value= 0.0, max_value = 200.0, value=1.0)
d = st.slider('Please insert your Blood Pressure',min_value= 0.0, max_value = 200.0, value=1.0)
e = st.slider('Please insert your Skin Thickness',min_value= 0.0, max_value = 100.0, value=1.0)
f = st.slider('Please insert your Insulin',min_value= 0.0, max_value = 1000.0, value=1.0)
g = st.slider('Please insert your Diabetes Pedigree Function',min_value= 0.0, max_value = 2.0, value=0.001)
h = st.slider('Please insert your Age',min_value= 1, max_value = 100, value=1)

heart_data = pd.read_csv(r'https://raw.githubusercontent.com/Ammarrosdi/diabetes/main/diabetes.csv')

import pandas as pd

diabetes = pd.read_csv('diabetes.csv')


X = diabetes.drop('Outcome',axis=1)

y = diabetes['Outcome']


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,random_state=1234,test_size=0.2)

diabetes.info()

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report


RandomForest = RandomForestClassifier()
RandomForest.fit(Xtrain, ytrain)
ypred = RandomForest.predict(Xtest)

print(confusion_matrix(ytest, ypred))
print()
print()
print(classification_report(ytest, ypred))



def prediction(BMI, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, DiabetesPedigreeFunction, Age):
    diabetes_data2 = pd.DataFrame(columns = ['BMI', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age' ])
    diabetes_data2 = diabetes_data2.append({'BMI' : BMI, 'Pregnancies' : Pregnancies, 'Glucose' : Glucose, 'BloodPressure' : BloodPressure, 'SkinThickness' : SkinThickness, 'Insulin' :Insulin, 'DiabetesPedigreeFunction' :DiabetesPedigreeFunction, 'Age' :Age}, ignore_index = True) 
    ypred = RandomForest.predict(diabetes_data2)
    st.write('Your prediction for have diabetes is:')
    if ypred ==1:
      st.write('Yes')
    else:
      st.write('No')  
    
prediction(a,b,c,d,e,f,g,h)
