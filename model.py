import pandas as pd
import numpy as np
import matplotlib.pyplot as presentingList
from sklearn import datasets, preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import sklearn.ensemble as ske
import time


class AdmitModel:

    # Accepted models
    modelList = {"NB": GaussianNB(),
                 "DT": DecisionTreeClassifier(),
                 "NN": KNeighborsClassifier(),
                 "LR": LogisticRegression(),
                 "SVM": SVC(),
                 "MP": MLPClassifier()
                 }

    # Builds modelType
    def __init__(self, modelType="DT"):

        # Check modelType is an acceptable modelType
        if modelType in self.modelList.keys():
            self.clf = self.modelList[modelType]
        else:
            print("Model type not recognised: Using default Decision Tree Classifier")
            self.clf = self.modelList["DT"]

        # Import Data
        df = pd.read_csv('elevenFeatures.csv', na_values=['NA'], dtype='category')
        df = df.dropna()

        # Drop columns for male, nursing home, indigenous, flg7days, business hours, weekend
        df = df[['triage_category', 'Ambulance', 'age_cat10', 'flgadmit30', 'present_sub_merged', 'admitted']]

        # Build Classifier
        x = df.drop(['admitted'], axis=1).values
        y = df['admitted'].values

        self.clf.fit(x, y)

    # Predict admission for new instance
    def calculate_admission(self, triage, ambulance, age, prevAdmit, presenting):
        # triage_category,Ambulance,age_cat10,male,NHome,indigenous,flg7days,flgadmit30,business_hours,weekend,present_sub_merged,admitted
        example = [[triage, ambulance, age, prevAdmit, presenting]]
        t0 = time.time()
        pr = self.clf.predict(example)
        t1 = time.time()
        print(t1-t0)
        return pr



# Receive 5 variables and classify - return yes or no
