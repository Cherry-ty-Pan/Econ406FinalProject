"""
Name: Cherry Pan
Class: Econ 406
Instructors: Phil Erickson, Konstantin Golyaev

This is the final project for Econ 406.
This project aims to better understand the impact of different variables on
heart disease, and to make a prediction on chances of getting heart disease
using logistic regression based on these variables/factors.

Data columns:
age: The person's age in years
sex: The person's sex(1 = male, 0 = female)
cp: chest pain type (0: asymptomatic, 1: atypical angina
					 2: non-anginal pain, 3: typical angina)
trestbps: The person's resting blood pressure (mmHg)
chol: The person's cholesterol measurement in mg/dl
fbs: The person's fasting blood suger(>120 mg/dl, 1 = true, 0 = false)
restecg: resting electrocardiographic results
		(0: showing probable or definite left ventricular hypertrophy by Estes'
           criteria
		 1: normal
		 2: having ST-T wave abormality(T wave inversions
		 	and/or ST levation or depression of >0.05 mV))
thalach: The person's maximum heart rate achieved
exang: Exercise induced angina (1 = yes, 0 = no)
oldpeak: WT depression induced by exercise relative to rest('ST' relates to
        positions on the ECG plot)
slope: the slope of the peak exercise ST segment
	   (0: downsloping, 1: flat, 2: upsloping)
ca: The number of major vessels (0-3)
thal: A blood disorder called thalassemia
	 (0: null (have already dropped from the dataset)
	  1: fixed defect(no blood flow in some part of the heart)
	  2: normal blood flow
	  3: reversible defect (a blood flow is observed but it is not normal))
target: heart disease (1 = no, 0 = yes)
"""

import statsmodels.api as sm
from sklearn.metrics import(confusion_matrix,
                            accuracy_score)
import pandas as pd
import seaborn as sns
from IPython import get_ipython
IPYTHON = get_ipython()
IPYTHON.magic('matplotlib')

def extract_data(file_name: str):
    """
    This function will import data from "heart.csv" and then clean data in
    which we remove all the missing value

    Parameter
    ---------
    file_name: str
        the dataset used to import data and clean data
    """
    dataframe = pd.read_csv(file_name)
    dataframe.dropna()
    return dataframe

def data_descriptive_statistics(file_name: str):
    """
    This function will generate some descriptive statistics (such as mean,
    mode, median) from the dataset

    Parameter
    ---------
    file_name: str
        the dataset we used for the descriptive statistics
    """
    dataframe = extract_data(file_name)
    print(dataframe.describe())

def data_graph(file_name: str):
    """
    This function will have the data visulization; it generates graphs
    that show the relationships between each factors and the chance having
    heart disease. We can tell what group of people or people with what kind
    of factor might have a greater chance of having heart disease.

    Parameter
    ---------
    file_name: str
        the dataset we used for the data visulization
    """
    dataframe = extract_data(file_name)
    sns.catplot(x='age', y='target', kind='bar', data=dataframe)
    sns.catplot(x='sex', y='target', kind='bar', data=dataframe)
    sns.catplot(x='cp', y='target', kind='bar', data=dataframe)
    sns.catplot(x='trestbps', y='target', kind='bar', data=dataframe)
    sns.catplot(x='chol', y='target', kind='bar', data=dataframe)
    sns.catplot(x='fbs', y='target', kind='bar', data=dataframe)
    sns.catplot(x='restecg', y='target', kind='bar', data=dataframe)
    sns.catplot(x='thalach', y='target', kind='bar', data=dataframe)
    sns.catplot(x='exang', y='target', kind='bar', data=dataframe)
    sns.catplot(x='oldpeak', y='target', kind='bar', data=dataframe)
    sns.catplot(x='slope', y='target', kind='bar', data=dataframe)
    sns.catplot(x='ca', y='target', kind='bar', data=dataframe)
    sns.catplot(x='thal', y='target', kind='bar', data=dataframe)


def data_model(file_name: str):
    """
    This function will generate a logistic regression analysis for the heart
    disease, and evaluate how accuarate the prediction is.

    Parameter
    ---------
    file_name: str
        the dataset we used to do the logistic regression
    """
    dataframe = extract_data(file_name)
    x_vari = dataframe[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                        'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
                        'ca', 'thal']]
    y_vari = dataframe['target']
    x_vari = sm.add_constant(x_vari)
    log_mod = sm.Logit(y_vari, x_vari)
    log_reg = log_mod.fit()
    print(log_reg.summary())
    yhat = log_reg.predict(x_vari)
    prediction = list(map(round, yhat))
    con_matrix = confusion_matrix(y_vari, prediction)
    print("Confusion Matrix : \n", con_matrix)
    print('Test accuracy = ', accuracy_score(y_vari, prediction))
