from typing import Type
import numpy as np
import pandas as pd
from sympy import true
import tensorflow as tf
import keras

##Przygotowanie modelu
def get_basic_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

##Wczytanie pliku danych
df = pd.read_csv("adult.data")

##Zamiana typow danych na kategorie, a następnie zakodowanie jako dane numeryczne w nowym DF
colsObjects = df.columns[df.dtypes == "object"].tolist()
for col in colsObjects:
    df[col] = df[col].astype('category')
    dfCoded = df
    dfCoded[col] = dfCoded[col].cat.codes
    dfCoded.loc[dfCoded[col] == -1, col] = np.nan

##Zamiana na float dla ujednolicenia typu
dfCoded = dfCoded.astype(np.float64)

print(dfCoded.head(30))

##Wybranie kolumny do wypełnienia
cols = dfCoded.columns.to_list()
print("Dostępne kolumny:")
print(cols)
while true:
    col = input("Etykieta kolumny do wypełnienia (aby anulować, wpisz \"koniec\"): ")
    if col in cols:
        print('Wpisano poprawnie')
        break
    elif col == 'koniec':
        exit()
    else:
        print('Nie ma takiej kolumny!')

##Podzielenie Dataframe na zawierające NaN w wybranej kolumnie i wypełnione
dfAllNan = dfCoded[dfCoded[col].isnull()]
dfNoNan = dfCoded[~dfCoded[col].isnull()]

dfNoNanTarget = dfNoNan.pop(col)
dfAllNanTarget = dfAllNan.pop(col)

model = get_basic_model()
model.fit(dfNoNan, dfNoNanTarget, epochs=15, batch_size=2)

