import streamlit as st 
import numpy as np 
import pandas as pd

#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report

st.title('Heart Disease Predictor')

st.sidebar.write("""
This is a web app demo using python libraries such as Streamlit, Sklearn etc
""")

st.sidebar.write ("For more info, please contact:")

st.sidebar.write("<a href='https://www.linkedin.com/in/mirulsraf/'>Amirul Asraf </a>", unsafe_allow_html=True)

a = st.slider('Please insert your BMI',min_value= 15.0, max_value = 50.0, value=0.2)
b = st.slider('Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] 1 for yes, 0 for no',min_value= 0, max_value = 1, value=1)
c = st.slider('Do you a heavy drinker? (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) 1 for yes, 0 for no',min_value= 0, max_value = 1, value=1)
d = st.slider('Do you have stroke? 1 for yes, 0 for no',min_value= 0, max_value = 1, value=1)
e = st.slider('Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? (0-30 days)',min_value= 0, max_value = 30, value=1)
f = st.slider('Thinking about your mental health, for how many days during the past 30 days was your mental health not good? (0-30 days)',min_value= 0, max_value = 30, value=1)
g = st.slider('Do you have serious difficulty walking or climbing stairs? 1 for yes, 0 for no',min_value= 0, max_value = 1, value=1)
h = st.slider('Do you have diabetes? 1 for yes, 0 for no',min_value= 0, max_value = 1, value=1)
i = st.slider('Do you have exercise during the past 30 days other than your regular jobs? 1 for yes, 0 for no',min_value= 0, max_value = 1, value=1)
j = st.slider('How you rate your general health? 1(bad) to 4(very good)',min_value= 0, max_value = 4, value=1)
k = st.slider('How long do you sleep in hours every day?',min_value= 0, max_value = 12, value=1)
l = st.slider('Do you have asthma? 1 for yes, 0 for no',min_value= 0, max_value = 1, value=1)
m = st.slider('Do you have kidney disease? 1 for yes, 0 for no',min_value= 0, max_value = 1, value=1)
n = st.slider('Do you have skin cancer? 1 for yes, 0 for no',min_value= 0, max_value = 1, value=1)

heart_data = pd.read_csv(r'https://raw.githubusercontent.com/mirulmio/HeartDisease/main/heart.csv')

import pandas as pd

diabetes = pd.read_csv('diabetes.csv')








X = diabetes.drop('Outcome',axis=1)

y = diabetes['Outcome']

from sklearn.model_selection import train_test_split
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

prediction(a,b,c,d,e,f,g,h,i,j,k,l,m,n)

def prediction(bmi, smoking, alcoholdrinking, stroke, physicalhealth, mentalhealth, diffwalking, diabetic, physicalactivity, genhealth, sleeptime, asthma, kidneydisease, skincancer):
    heart_data2 = pd.DataFrame(columns = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Diabetic', 'PhysicalActivity',  'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer' ])
    heart_data2 = heart_data2.append({'BMI' : bmi, 'Smoking' : smoking, 'AlcoholDrinking' : alcoholdrinking, 'Stroke' : stroke, 'PhysicalHealth' : physicalhealth, 'MentalHealth' :mentalhealth, 'DiffWalking' :diffwalking, 'Diabetic' :diabetic, 'PhysicalActivity' :physicalactivity,  'GenHealth' :genhealth, 'SleepTime' :sleeptime, 'Asthma' :asthma, 'KidneyDisease' :kidneydisease, 'SkinCancer' :skincancer}, ignore_index = True) 
    ypred = nb.predict(heart_data2)
    st.write('Your prediction for have heart disease is:')
    if ypred ==1:
      st.write('Yes')
    else:
      st.write('No')  
