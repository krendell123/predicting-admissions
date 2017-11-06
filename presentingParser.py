import pandas as pd
import numpy as np

# Create a new dictionary (presenting_dictionary) which will hold all the dictionaries to presenting problem sub categories
presenting_list = []
for i in range(0,20):
    sub_dict = {}
    presenting_list.append(sub_dict)

# Read in presenting problem file
df = pd.read_csv('PresentingProblemList.csv', na_values=['NA'], dtype='category')
df = df.dropna()

for row in df.itertuples():
    index = int(row.DESTINY) - 1
    if index in range(0,20):
        presenting_list[index][row.KEYWORDS] = row.FINAL
