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

def wybory(): ##Funkcje wyboru
    print("Aby wyjśćz programu, wpisz \"koniec\".")

    df = wybor_pliku()
    col = wybor_kolumny(df)

    return df, col

def wybor_pliku():  ##Wybranie i wczytanie pliku do pracy

    while true:
        file = input("Wpisz nazwę pliku lub wciśnij Enter aby wybrać domyślny: ")
        if file == None:
            print("Wybrano domyślny plik adult.data")
            df = pd.read_csv("adult.data")
            break
        elif file == 'koniec':
            exit()
        else:
            print("Wybrano plik " + file)
            try:
                df = pd.read_csv(file)
                break
            except:
                print("Wpisano niepoprawną nazwę pliku, proszę upewnić się czy plik znajduje się w folderze programu.")

    return df

def wybor_kolumny(df):    ##Wybranie kolumny do wypełnienia
    
    cols = df.columns.to_list()
    print("Dostępne kolumny:")
    print(cols)
    while true:
        col = input("Etykieta kolumny do wypełnienia: ")
        if col in cols:
            print('Wybrano poprawną kolumnę ' + col)
            break
        elif col == 'koniec':
            exit()
        else:
            print('Nie ma takiej kolumny!')

    return col

def przygotowanie_danych(): ##Główna funkcja przygotowująca dane do nauki modelu
    
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
df, col = wybory()

dfAllNan, dfNoNan, dfAllNanTarget, dfNoNanTarget = przygotowanie_danych()

model = get_basic_model()
model.fit(dfNoNan, dfNoNanTarget, epochs=15, batch_size=2)