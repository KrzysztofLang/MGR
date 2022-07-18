from typing import Type
import numpy as np
import pandas as pd
from sympy import true
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import svm
from math import sqrt
import re


# Funkcje wyboru
def wybory():
    print('Aby wyjść z programu, wpisz "koniec".\n')

    df = wybor_pliku()
    col = wybor_kolumny(df)

    return df, col


# Wybranie i wczytanie pliku do pracy
def wybor_pliku():
    while true:
        file = input(
            "Wpisz nazwę pliku lub wciśnij Enter aby wybrać domyślny: "
        )

        if not file:
            print("Wybrano domyślny plik adult.data\n")
            df = pd.read_csv("adult.data")
            break
        elif file == "koniec":
            exit()
        else:
            try:
                df = pd.read_csv(file)
                print("Wybrano plik " + file + "\n")
                break
            except:
                print("Wpisano niepoprawną nazwę pliku, proszę upewnić się")
                print("czy plik znajduje się w folderze programu.\n")

    return df


# Wybranie kolumny do wypełnienia
def wybor_kolumny(df):
    cols = df.columns.to_list()
    print("Dostępne kolumny:")
    print(cols)
    while true:
        col = input(
            "Wpisz nazwę kolumny lub wciśnij Enter aby wybrać domyślną: "
        )
        if not col:
            print("Wybrano domyślną kolumnę workclass\n")
            col = "workclass"
            break
        elif col in cols:
            print("Wybrano poprawną kolumnę " + col + "\n")
            break
        elif col == "koniec":
            exit()
        else:
            print("Nie ma takiej kolumny!\n")

    return col


# Główna funkcja przygotowująca dane do nauki modelu
def przygotowanie_danych(df, col):
    # Zamiana typow danych na kategorie, a następnie zakodowanie jako dane
    # numeryczne w nowym DF
    cols_objects = df.columns[df.dtypes == "object"].tolist()
    for cols in cols_objects:
        df[cols] = df[cols].astype("category")

    df = pd.get_dummies(df, dummy_na=True)
    print(df)

    nan_df = df.loc[:, df.columns.str.endswith("_nan")]
    print(nan_df)

    pattern = "^([^_]*)_"
    regex = re.compile(pattern)

    for index in df.index:
        for col_nan in nan_df.columns:
            if df.loc[index, col_nan] == 1:
                col_id = regex.search(col_nan).group(1)
                targets = df.columns[df.columns.str.startswith(col_id + "_")]
                df.loc[index, targets] = np.nan

    df.drop(df.columns[df.columns.str.endswith("_nan")], axis=1, inplace=True)

    df = df.astype(np.float64)

    targets = df.columns[df.columns.str.startswith(col + "_")]

    df = df[
        [c for c in df if c not in targets] + [c for c in targets if c in df]
    ]
    print(df)
    print(df.info())
    cat_num = len(targets)

    # Podzielenie Dataframe na zawierające NaN w wybranej kolumnie i
    # wypełnione
    for index in targets:
        df_all_nan = df[df[index].isnull()]
        df_no_nan = df[~df[index].isnull()]

    print(df_all_nan.info())
    print(df_no_nan.info())

    return df_all_nan, df_no_nan, cat_num


# Wczytanie pliku danych
df, col = wybory()

df_all_nan, df_no_nan, cat_num = przygotowanie_danych(df, col)


clf = svm.SVC(kernel='linear')