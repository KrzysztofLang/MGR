
import numpy as np
import pandas as pd
s = pd.read_csv("adult.data")
#print(s.info())
#unique_counts = pd.DataFrame.from_records([(col, s[col].nunique()) for col in s.columns],
#                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])
#print(unique_counts)

######Zamiana typow danych na kategorie
cols_to_include = ['workclass', 'education', 'martial-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'attribute']
for col in s.columns:
    if col in cols_to_include:
        s[col] = s[col].astype('category')

print(s.info())
print(s)
print(s.iloc[1])