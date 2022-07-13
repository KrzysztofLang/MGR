from typing import Type
import numpy as np
import pandas as pd
from sympy import true
import tensorflow as tf
import keras


# Przygotowanie modelu
def get_basic_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


# Funkcje wyboru
def wybory():
    print('Aby wyjść z programu, wpisz "koniec".')

    df = wybor_pliku()
    col = wybor_kolumny(df)

    return df, col


# Wybranie i wczytanie pliku do pracy
def wybor_pliku():
    while true:
        file = input(
            "Wpisz nazwę pliku lub wciśnij Enter aby wybrać \
            domyślny: "
        )

        if not file:
            print("Wybrano domyślny plik adult.data")
            df = pd.read_csv("adult.data")
            break
        elif file == "koniec":
            exit()
        else:
            print("Wybrano plik " + file)
            try:
                df = pd.read_csv(file)
                break
            except:
                print(
                    "Wpisano niepoprawną nazwę pliku, proszę upewnić się \
                    czy plik znajduje się w folderze programu."
                )

    return df


# Wybranie kolumny do wypełnienia
def wybor_kolumny(df):
    cols = df.columns.to_list()
    print("Dostępne kolumny:")
    print(cols)
    while true:
        col = input("Etykieta kolumny do wypełnienia: ")
        if col in cols:
            print("Wybrano poprawną kolumnę " + col)
            break
        elif col == "koniec":
            exit()
        else:
            print("Nie ma takiej kolumny!")

    return col


# Główna funkcja przygotowująca dane do nauki modelu
def przygotowanie_danych():
    # Zamiana typow danych na kategorie, a następnie zakodowanie jako dane
    # numeryczne w nowym DF
    cols_objects = df.columns[df.dtypes == "object"].tolist()
    for col in cols_objects:
        df[col] = df[col].astype("category")
        df_coded = df
        df_coded[col] = df_coded[col].cat.codes
        df_coded.loc[df_coded[col] == -1, col] = np.nan

    # Zamiana na float dla ujednolicenia typu
    df_coded = df_coded.astype(np.float64)

    # Podzielenie Dataframe na zawierające NaN w wybranej kolumnie i
    # wypełnione
    df_all_nan = df_coded[df_coded[col].isnull()]
    df_no_nan = df_coded[~df_coded[col].isnull()]

    # Wydzielenie danych uczących
    df_no_nan_target = df_no_nan.pop(col)
    df_all_nan_target = df_all_nan.pop(col)

    return df_all_nan, df_no_nan, df_all_nan_target, df_no_nan_target


# Wczytanie pliku danych
df, col = wybory()

(
    df_all_nan,
    df_no_nan,
    df_all_nan_target,
    df_no_nan_target,
) = przygotowanie_danych()

model = get_basic_model()
model.fit(df_no_nan, df_no_nan_target, epochs=15, batch_size=2)
