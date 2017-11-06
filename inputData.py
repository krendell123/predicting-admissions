import pandas as pd
import numpy as np

triage = {
    "1":"1.0",
    "2":"2.0",
    "3":"3.0",
    "4":"4.0",
    "5":"5.0"
}

boolean = {
    "Yes":"1",
    "No":"0"
}

gender = {
    "Male":"1",
    "Female":"0"
}

age = {
    "16-19":"1",
    "20-39":"2",
    "40-59":"3",
    "60-79":"4",
    "80+":"5"
}

hour = {
    "08:00-16:59":"1",
    "17:00-23:59":"2",
    "00:00-07:59":"0"
}

presenting = {
    "Gastrointestinal":1,
    "Cardiovascular":2,
    "Unwell":3,
    "Infection":4,
    "Injury": 5,
    "Respiratory":6,
    "Muscular Skeletal": 9,
    "Neurological": 10,
    "Mental Health": 11,
    "Toxicity": 12,
    "ENT": 13,
    "Administration": 14,
    "Urinology": 15,
    "Social": 16,
    "Endocrine": 17,
    "Obstetrics/Gynaecology": 18,
    "Allergy/Skin": 19,
    "Haematology/Oncology": 20
}

# Create a new list (presenting_list) which will hold all the dictionaries of presenting problem subcategories
presenting_list = []
for i in range(0,20):
    sub_dict = {}
    presenting_list.append(sub_dict)

# Read in presenting problem file which contains all presenting problem keywords and encodings
df = pd.read_csv('PresentingProblemCropped.csv', na_values=['NA'], dtype='category')
df = df.dropna()

# Loop through rows and add each subcategory to the appropriate dictionary in the list
for row in df.itertuples():
    index = int(row.DESTINY) - 1
    if index in range(0,20):
        presenting_list[index][row.KEYWORDS] = row.FINAL
