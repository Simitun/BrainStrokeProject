import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
#import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

st.title('Brain Stroke Prediction')
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp{{
            background-image: url("https://www.jaipurneuro.com/wp-content/uploads/2019/03/4.png");
            background-attachment: fixed;
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html = True
    )
add_bg_from_url()

BrainStroke = pd.read_csv('healthcare-dataset-stroke-data.csv')

BrainStroke.dropna(inplace = True)

CheckData = st.sidebar.checkbox('Brain Stroke Data')
if CheckData:
    st.write(BrainStroke)
    
del BrainStroke['id']    

encoder = LabelEncoder()
BrainStroke['gender'] = encoder.fit_transform(BrainStroke['gender'])
gender = {index : label for index, label in enumerate(encoder.classes_)}
#gender

BrainStroke['ever_married'] = encoder.fit_transform(BrainStroke['ever_married'])
ever_married = {index : label for index, label in enumerate(encoder.classes_)}
#ever_married

BrainStroke['work_type'] = encoder.fit_transform(BrainStroke['work_type'])
work_type = {index : label for index, label in enumerate(encoder.classes_)}
#work_type

BrainStroke['Residence_type'] = encoder.fit_transform(BrainStroke['Residence_type'])
Residence_type = {index : label for index, label in enumerate(encoder.classes_)}
#Residence_type

BrainStroke['smoking_status'] = encoder.fit_transform(BrainStroke['smoking_status'])
smoking_status = {index : label for index, label in enumerate(encoder.classes_)}
#smoking_status

# '''For Checkbox'''
CheckEncodeData = st.sidebar.checkbox('Brain Stroke Encode Data')
if CheckEncodeData:
    st.write(BrainStroke)
# name = 'Correlation Matrix'
# Name = st.sidebar.checkbox('Matrix Name')
# if Name:
#     st.write(name)
    
# import matplotlib.pyplot as plt    
# from matplotlib import pyplot as plt
# import seaborn as sns    
# sns.heatmap(BrainStroke.corr(),annot = True, cmap = 'RdYlGn', linewidths = 0.2)
# fig = plt.gcf()
# fig.set_size_inches(10,8)
# plt.show()
# Connection = st.sidebar.checkbox('Correlation Matrix')
# if Connection:
#     st.write(fig)    

# '''Dividing Feature and Label'''
x = BrainStroke.iloc[: ,  :10]
y = BrainStroke.iloc[: , 10 : 11]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# '''Under Sampling'''
Stroke = BrainStroke[BrainStroke.stroke == 1]
NoStroke = BrainStroke[BrainStroke.stroke == 0]

NoStrokeSample = NoStroke.sample(n = 50)
BrainStroke = pd.concat([NoStrokeSample, Stroke], axis = 0)
#NoStrokeSample = NoStroke.sample(n = 50)
    
# '''User Input'''
def InputUser():
             
    col1, col2 = st.columns(2)
    with col1:
        Gender = st.selectbox('Gender', ('Female', 'Male', 'Other'))
        if Gender == 'Female':
            gender = 0
        elif Gender == 'Male':
            gender = 1
        else:
            gender = 2
    with col2:
        Age = st.number_input('Age: Please enter from 0 to 82', min_value = 0.00, max_value = 82.00, value = 45.00, step = 0.5)
        
    col1, col2 = st.columns(2)        
    with col1:
        Hypertension = st.selectbox('Hypertension(High Blood Pressure)', ('No', 'Yes'))
        if Hypertension == 'No':
            hypertension = 0
        else:
            hypertension = 1        
    with col2:
        HeartDisease = st.selectbox('Heart Disease', ('No', 'Yes'))
        if HeartDisease == 'No':
            heart_disease = 0
        else:
            heart_disease = 1            
  
    col1, col2 = st.columns(2)    
    with col1:
        Ever_Married = st.selectbox('Ever Married', ('No', 'Yes'))
        if Ever_Married == 'No':
            ever_married = 0
        else:
            ever_married = 1
    with col2:
        Work_Type = st.selectbox('Work Type', ('Government Job', 'Never Worked', 'Private', 'Self-employed', 'Children'))
        if Work_Type == 'Government Job':
            work_type = 0
        elif Work_Type == 'Never Worked':
            work_type = 1
        elif Work_Type == 'Pirvate':
            work_type = 2
        elif Work_Type == 'Self-employed':
            work_type = 3
        else:
            work_type = 4            
        
    col1, col2 = st.columns(2)        
    with col1:
        Residence = st.selectbox('Residence Type', ('Rural', 'Urban'))
        if Residence == 'Rural':
            residence = 0
        else:
            residence = 1
    with col2:
        Avg_Glu_Level = st.number_input('Average Glucose Level: Please enter from 55 to 272', min_value = 55.00, max_value = 272.00, value = 150.00, step = 0.5)
        
    col1, col2 = st.columns(2)        
    with col1:
        BMI = st.number_input('BMI(Body Mass Index): Please enter from 10 to 98', min_value = 10.00, max_value = 98.00, value = 72.00, step = 0.5)
    with col2:
        Smoking_Status = st.selectbox('Smoking Status', ('Unknown', 'Formely Smoked', 'Never Smoked', 'Smokes'))
        if Smoking_Status == 'Unknown':
            smoking_status = 0
        elif Smoking_Status == 'Formely Smoked':
            smoking_status = 1
        elif Smoking_Status == 'Never Smoked':
            smoking_status = 2
        else:
            smoking_status = 3        
        
    Data = {'Gender' : gender, 
            'Hypertension' : hypertension, 
            'HeartDisease' : heart_disease, 
            'Ever_Married' : ever_married, 
            'Work_Type' : work_type,
            'Residence' : residence, 
            'Avg_Glu_Level' : Avg_Glu_Level, 
            'BMI' : BMI, 
            'Smoking_Status' : smoking_status,
            'Age' : Age}
    features = pd.DataFrame(Data, index = [0])
    return features

classifier = st.sidebar.selectbox('Choose model classifier', ('Logistic Regression', 'Naive Bayes', 'K-Nearest Neighbors', 'Support Vector Machine', 'Decision Tree Classifier', 'Random Forest Classifier', 'XGB Classifier'))

# '''Modeling'''
if classifier == 'Logistic Regression':
    st.subheader('Using Logistic Regression Classifier')
    from sklearn.linear_model import LogisticRegression
    LR_model = LogisticRegression()
    LR_model.fit(x_train, y_train)
    
#     '''Evaluation'''
    x_train_pred = LR_model.predict(x_train)
    x_test_pred = LR_model.predict(x_test)
    
    input_df = InputUser()
    test_button = st.button('Check result')
    if test_button:
        input_data_as_numpy_array = np.asarray(input_df)
        input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
        std_data =scaler.transform(input_data_reshaped)
        
        prediction = LR_model.predict(std_data)
        if prediction[0] == 1:
            st.warning('This person is at risk of brain stroke. Be Careful!', icon = "⚠️")
        if prediction[0] == 0:
            st.success('This person is not at risk of brain stroke.', icon = "✅")
            
#'''Naive Bayes'''         
if classifier == 'Naive Bayes':
        st.subheader('Using Naive Bayes Classifier')
        from sklearn.naive_bayes import GaussianNB
        NB_model = GaussianNB()
        NB_model.fit(x_train, y_train)
        
        x_train_pred = NB_model.predict(x_train)
        x_test_pred = NB_model.predict(x_test)
    
        input_df = InputUser()
        test_button = st.button('Check result')
        if test_button:
            input_data_as_numpy_array = np.asarray(input_df)
            input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
            std_data =scaler.transform(input_data_reshaped)
        
            prediction = NB_model.predict(std_data)
            if prediction[0] == 1:
                st.warning('This person is at risk of brain stroke. Be Careful!', icon = "⚠️")
            if prediction[0] == 0:
                st.success('This person is not at risk of brain stroke.', icon = "✅")
            
#'''K-Nearest Neighbors'''         
if classifier == 'K-Nearest Neighbors':
        st.subheader('Using K-Nearest Neighbors Classifier')
        from sklearn.neighbors import KNeighborsClassifier
        KNN_model = KNeighborsClassifier(n_neighbors = 13)
        KNN_model.fit(x_train, y_train)
    
        x_train_pred = KNN_model.predict(x_train)
        x_test_pred = KNN_model.predict(x_test)
    
        input_df = InputUser()
        test_button = st.button('Check result')
        if test_button:
            input_data_as_numpy_array = np.asarray(input_df)
            input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
            std_data =scaler.transform(input_data_reshaped)
        
            prediction = KNN_model.predict(std_data)
            if prediction[0] == 1:
                st.warning('This person is at risk of brain stroke. Be Careful!', icon = "⚠️")           
            if prediction[0] == 0:
                st.success('This person is not at risk of brain stroke.', icon = "✅")
    
#'''Support Vector Machine'''
if classifier == 'Support Vector Machine':
        st.subheader('Using Support Vector Machine Classifier')
        from sklearn.svm import SVC
        SVM_model = SVC(gamma = 'auto')
        SVM_model.fit(x_train, y_train)
    
        x_train_pred = SVM_model.predict(x_train)
        x_test_pred = SVM_model.predict(x_test)
    
        input_df = InputUser()
        test_button = st.button('Check result')
        if test_button:
            input_data_as_numpy_array = np.asarray(input_df)
            input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
            std_data =scaler.transform(input_data_reshaped)
        
            prediction = SVM_model.predict(std_data)
            if prediction[0] == 1:
                st.warning('This person is at risk of brain stroke. Be Careful!', icon = "⚠️")                
            if prediction[0] == 0:
                st.success('This person is not at risk of brain stroke.', icon = "✅")
    
#'''Decision Tree Classifier'''         
if classifier == 'Decision Tree Classifier':
    st.subheader('Using Decision Tree Classifier')
    from sklearn import tree 
    DTC_model = tree.DecisionTreeClassifier()
    DTC_model.fit(x_train, y_train)
    
    x_train_pred = DTC_model.predict(x_train)
    x_test_pred = DTC_model.predict(x_test)
    
    input_df = InputUser()
    test_button = st.button('Check result')
    if test_button:
        input_data_as_numpy_array = np.asarray(input_df)
        input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
        std_data =scaler.transform(input_data_reshaped)
        
        prediction = DTC_model.predict(std_data)
        if prediction[0] == 1:
            st.warning('This person is at risk of brain stroke. Be Careful!', icon = "⚠️")            
        if prediction[0] == 0:
            st.success('This person is not at risk of brain stroke.', icon = "✅")
    
#'''Random Forest Classifier'''         
if classifier == 'Random Forest Classifier':
    st.subheader('Using Random Forest Classifier')
    from sklearn.ensemble import RandomForestClassifier 
    RFC_model = RandomForestClassifier(n_estimators = 10)
    RFC_model.fit(x_train, y_train)
    
    x_train_pred = RFC_model.predict(x_train)
    x_test_pred = RFC_model.predict(x_test)
    
    input_df = InputUser()
    test_button = st.button('Check result')
    if test_button:
        input_data_as_numpy_array = np.asarray(input_df)
        input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
        std_data =scaler.transform(input_data_reshaped)
        
        prediction = RFC_model.predict(std_data)
        if prediction[0] == 1:
            st.warning('This person is at risk of brain stroke. Be Careful!', icon = "⚠️")            
        if prediction[0] == 0:
            st.success('This person is not at risk of brain stroke.', icon = "✅")
    
#'''XGB Classifier'''         
if classifier == 'XGB Classifier':
    st.subheader('Using XGB Classifier')
    from xgboost import XGBClassifier
    XGB_model = XGBClassifier(objective = 'binary:logistic', learning_rate = 0.2)
    XGB_model.fit(x_train, y_train)
    
    x_train_pred = XGB_model.predict(x_train)
    x_test_pred = XGB_model.predict(x_test)
    
    input_df = InputUser()
    test_button = st.button('Check result')
    if test_button:
        input_data_as_numpy_array = np.asarray(input_df)
        input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1)
        std_data =scaler.transform(input_data_reshaped)
        
        prediction = XGB_model.predict(std_data)
        if prediction[0] == 1:
            st.warning('This person is at risk of brain stroke. Be Careful!', icon = "⚠️")
        if prediction[0] == 0:
            st.success('This person is not at risk of brain stroke.', icon = "✅")

# Title = 'Experiment'
# title = st.sidebar.checkbox('Experiment')
# if title:
#     st.write(Title)
            
ExpModels = pd.DataFrame({'Name of Models' : ['Logistic Regression', 'Naive Bayes Classifier', 'K-Nearest Neighbors', 'Support Vector Machine', 'Decision Tree Classifier', 'Random Forest Classifier', 'XGB Classifier'], 'Accuracy (%)' : [96.3340, 86.8296, 96.2661, 96.1982, 91.8534, 96.1982, 96.0625], 'Precision (%)' : [98.1658, 56.5565, 48.1331, 48.1318, 53.2501, 68.1948, 66.4376], 'Recall (%)' : [50.9091, 71.3136, 50.0000, 49.9647, 54.6987, 51.7124, 53.3895], 'F1 (%)' : [50.8515, 58.2078, 49.0488, 49.0311, 53.7471, 52.3631, 55.0537], 'Comfusion Matrix' : [[[1179, 0], [49, 0]], [[1044, 135], [24, 25]], [[1178, 1], [49, 0]], [[1179, 0], [49, 0]], [[1113, 66], [40, 9]], [[1177, 2], [49, 0]], [[1176, 3], [47, 2]]]})

ExpScores = st.sidebar.checkbox('Experiment')
if ExpScores:
    st.write(ExpModels)    

    
# cm_name = st.sidebar.selectbox('Choose Model Name', ('Logistic Regression', 'Naive Bayes', 'K-Nearest Neighbors', 'Support Vector Machine',  'Decision Tree Classifier', 'Random Forest Classifier', 'XGB Classifier'))
# if cm_name == 'Logistic Regression':
#     from sklearn.linear_model import LogisticRegression
#     LR_model = LogisticRegression()
#     LR_model.fit(x_train, y_train)
#     y_pred = LR_model.predict(x_test)
#     from sklearn.metrics import confusion_matrix
#     ConfusionMatrix = confusion_matrix(y_test, y_pred)
#     st.text("Confusion Matrix for Logistic Regression")
#     st.write(ConfusionMatrix)
    
# if cm_name == 'Naive Bayes':
#     from sklearn.naive_bayes import GaussianNB
#     NB_model = GaussianNB()
#     NB_model.fit(x_train, y_train)
#     y_pred = NB_model.predict(x_test)
#     from sklearn.metrics import confusion_matrix
#     ConfusionMatrix = confusion_matrix(y_test, y_pred)
#     st.text("Confusion Matrix for Naive Bayes")
#     st.write(ConfusionMatrix)
    
# if cm_name == 'K-Nearest Neighbors':
#     from sklearn.neighbors import KNeighborsClassifier
#     KNN_model = KNeighborsClassifier(n_neighbors = 13)
#     KNN_model.fit(x_train, y_train)
#     y_pred = KNN_model.predict(x_test)
#     from sklearn.metrics import confusion_matrix
#     ConfusionMatrix = confusion_matrix(y_test, y_pred)
#     st.text("Confusion Matrix for K-Nearest Neighbors")
#     st.write(ConfusionMatrix)
    
# if cm_name == 'Support Vector Machine':
#     from sklearn.svm import SVC
#     SVM_model = SVC(gamma = 'auto')
#     SVM_model.fit(x_train, y_train)
#     y_pred = SVM_model.predict(x_test)
#     from sklearn.metrics import confusion_matrix
#     ConfusionMatrix = confusion_matrix(y_test, y_pred)
#     st.text("Confusion Matrix for Support Vector Machine")
#     st.write(ConfusionMatrix)
    
# if cm_name == 'Decision Tree Classifier':
#     from sklearn import tree 
#     DTC_model = tree.DecisionTreeClassifier()
#     DTC_model.fit(x_train, y_train)
#     y_pred = DTC_model.predict(x_test)
#     from sklearn.metrics import confusion_matrix
#     ConfusionMatrix = confusion_matrix(y_test, y_pred)
#     st.text("Confusion Matrix for Decision Tree Classifier")
#     st.write(ConfusionMatrix)
    
# if cm_name == 'Random Forest Classifier':
#     from sklearn.ensemble import RandomForestClassifier 
#     RFC_model = RandomForestClassifier(n_estimators = 10)
#     RFC_model.fit(x_train, y_train)
#     y_pred = RFC_model.predict(x_test)
#     from sklearn.metrics import confusion_matrix
#     ConfusionMatrix = confusion_matrix(y_test, y_pred)
#     st.text("Confusion Matrix for Random Forest Classifier")
#     st.write(ConfusionMatrix)
    
# if cm_name == 'XGB Classifier':
#     from xgboost import XGBClassifier
#     XGB_model = XGBClassifier(objective = 'binary:logistic', learning_rate = 0.2)
#     XGB_model.fit(x_train, y_train)
#     y_pred = XGB_model.predict(x_test)
#     from sklearn.metrics import confusion_matrix
#     ConfusionMatrix = confusion_matrix(y_test, y_pred)
#     st.text("Confusion Matrix for XGB Classifier")
#     st.write(ConfusionMatrix)


datas = st.sidebar.selectbox('About Health Knowledge', ('Causing Facts', 'Preventing Facts', 'Symptoms'))
if datas == 'Causing Facts':
    #st.metric(label = 'Causing Facts!', value = 'Hypertension (high blood pressure), Heart Disease, Diabetes')
    Facts = 'Causing Facts  =>  Hypertension (high blood pressure), Heart Disease, Diabetes'
    st.write(Facts)

if datas == 'Preventing Facts':    
    #st.metric(label = "Pervention Facts!", value = "Choose healthy foods and drinks, Keep a healthy weight, Get regular physical activity, Don't smoke, Limit alcohol, Check cholesterol, Control blood pressure, Control diabetes")
    Facts = "Preventing Facts  =>  Choose healthy foods and drinks, Keep a healthy weight, Get regular physical activity, Don't smoke, Limit alcohol, Check cholesterol, Control blood pressure, Control diabetes"
    st.write(Facts)

if datas == 'Symptoms':
    #st.metric(label = 'Symptoms!', value = 'Sudden numbness or weakness in the face, arm or leg (especially on one side of the body), Sudden vision problems in one or both eyes, Severe headache with no known cause')
    Facts = 'Symptoms  =>  Sudden numbness or weakness in the face, arm or leg (especially on one side of the body), Sudden vision problems in one or both eyes, Severe headache with no known cause'
    st.write(Facts)    





















