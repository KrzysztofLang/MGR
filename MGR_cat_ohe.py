from typing import Type
import numpy as np
import pandas as pd
from sympy import true
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import HistGradientBoostingClassifier
from math import sqrt
import re


class Dane:
    def __init__(self) -> None:
        
        print('Aby wyjść z programu, wpisz "koniec".\n')
        self.df = self.wybor_pliku()
        self.col = self.wybor_kolumny(self.df)
        self.col_type = self.df[self.col].dtype

    # Wybranie i wczytanie pliku do pracy
    @staticmethod
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

    @staticmethod
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


def przygotowanie_danych_kategoryczne(dane):
    # Zamiana typow danych na kategorie, a następnie zakodowanie jako dane
    # numeryczne w nowym DF
    col = dane.col
    df = dane.df

    cols_objects = df.columns[df.dtypes == "object"].tolist()
    for cols in cols_objects:
        df[cols] = df[cols].astype("category")

    df[col] = df[col].cat.codes
    df.loc[df[col] == -1, col] = np.nan

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

    df_temp = df.pop(col)
    df.insert(0, col, df_temp)

    # Podzielenie Dataframe na zawierające NaN w wybranej kolumnie i
    # wypełnione
    df_all_nan = df[df[col].isnull()]
    df_no_nan = df[~df[col].isnull()]

    return df_all_nan, df_no_nan


def naucz_model(df):
    # Zamiana DataFrame na array
    df = df.to_numpy()

    # Wydzielenie zbiorów uczących i testowych
    x_train, x_test, y_train, y_test = train_test_split(
        df[:, 1:], df[:, 0], test_size=0.3, random_state=109
    )

    # Nauka modelu
    clf = HistGradientBoostingClassifier(max_iter=100).fit(x_train, y_train)

    # Test skuteczności modelu
    y_pred = clf.predict(x_test)
    print("Skuteczność: ", metrics.accuracy_score(y_test, y_pred))


dane = Dane()

df_all_nan, df_no_nan = przygotowanie_danych_kategoryczne(dane)

naucz_model(df_no_nan)
