from typing import Type
import numpy as np
import pandas as pd
from sympy import true
import tensorflow as tf
import keras


def get_basic_model():  ##Przygotowanie modelu
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return model

def wybor_kolumny():    ##Wybranie kolumny do wypełnienia
    
    cols = df.columns.to_list()
    print("Dostępne kolumny:")
    print(cols)
    while true:
        col = input("Etykieta kolumny do wypełnienia (aby anulować, wpisz \"koniec\"): ")
        if col in cols:
            print('Wybrano poprawną kolumnę ' + col)
            break
        elif col == 'koniec':
            exit()
        else:
            print('Nie ma takiej kolumny!')

    return col

def przygotowanie_danych():
    
    ##Zamiana typow danych na kategorie, a następnie zakodowanie jako dane numeryczne w nowym DF
    colsObjects = df.columns[df.dtypes == "object"].tolist()
    for col in colsObjects:
        df[col] = df[col].astype('category')
        dfCoded = df
        dfCoded[col] = dfCoded[col].cat.codes
        dfCoded.loc[dfCoded[col] == -1, col] = np.nan

    ##Zamiana na float dla ujednolicenia typu
    dfCoded = dfCoded.astype(np.float64)

    ##Podzielenie Dataframe na zawierające NaN w wybranej kolumnie i wypełnione
    dfAllNan = dfCoded[dfCoded[col].isnull()]
    dfNoNan = dfCoded[~dfCoded[col].isnull()]

    ##Wydzielenie danych uczących
    dfNoNanTarget = dfNoNan.pop(col)
    dfAllNanTarget = dfAllNan.pop(col)

    return dfAllNan, dfNoNan, dfAllNanTarget, dfNoNanTarget

##Wczytanie pliku danych
df = pd.read_csv("adult.data")

col = wybor_kolumny()

dfAllNan, dfNoNan, dfAllNanTarget, dfNoNanTarget = przygotowanie_danych()

model = get_basic_model()
model.fit(dfNoNan, dfNoNanTarget, epochs=15, batch_size=2)