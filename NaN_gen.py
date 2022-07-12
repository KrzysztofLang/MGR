from typing import Type
import numpy as np
import pandas as pd

##Wczytanie pliku danych
df = pd.read_csv("indexData_NYA.csv")

print(df.info())