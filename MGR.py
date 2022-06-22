from typing import Type
import numpy as np
import pandas as pd
from sympy import true
import tensorflow as tf
import keras

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
""" print(df.info())
print(df) """

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



##Przeniesienie wybranej kolumny na koniec
""" cols.sort(key = col.__eq__)
dfCoded = dfCoded[cols] """

##Podzielenie Dataframe na zawierające NaN w wybranej kolumnie i wypełnione
dfAllNan = dfCoded[dfCoded[col].isnull()]
dfNoNan = dfCoded[~dfCoded[col].isnull()]

dfNoNanTarget = dfNoNan.pop(col)
dfAllNanTarget = dfAllNan.pop(col)

model = get_basic_model()
model.fit(dfNoNan, dfNoNanTarget, epochs=15, batch_size=2)

""" print(dfNoNan)
print(dfNoNanTarget) """

##Rozdzielenie na dane wejściowe i wyjściowe
""" x_df_no_nan = df_no_nan[:,0:13]
y_df_no_nan = df_no_nan[:,13]

print(x_df_no_nan)
print(y_df_no_nan) """
