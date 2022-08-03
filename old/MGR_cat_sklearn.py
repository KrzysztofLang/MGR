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
    """Pola"""

    def __init__(self) -> None:

        print('Aby wyjść z programu, wpisz "koniec".\n')
        self.df = self.wybor_pliku()
        self.col = self.wybor_kolumny(self.df)
        col_type = self.df[self.col].dtype

        match col_type:
            case "object" | "category":
                (
                    self.features_all_nan,
                    self.target_all_nan,
                    self.features_no_nan,
                    self.target_no_nan,
                    self.cat_arr,
                ) = self.przygotowanie_danych_kategoryczne()
            case _:
                raise ValueError("Nieobsługiwany typ danych do wypełnienia")

    # Wybranie i wczytanie pliku do pracy
    @staticmethod
    def wybor_pliku():
        while true:
            file = input(
                "Wpisz nazwę pliku lub wciśnij Enter aby wybrać domyślny: "
            )

            if not file:
                print("Wybrano domyślny plik wrkcls_occ.csv\n")
                df = pd.read_csv("wrkcls_occ.csv")
                break
            elif file == "koniec":
                exit()
            else:
                try:
                    df = pd.read_csv(file)
                    print("Wybrano plik " + file + "\n")
                    break
                except Exception:
                    print(
                        "Wpisano niepoprawną nazwę pliku, proszę upewnić się"
                    )
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

    # Przygotowanie danych do dalszej pracy w wypadku gdy
    # Wybrana kolumna zawiera dane kategoryczne
    def przygotowanie_danych_kategoryczne(self):
        # Zamiana typow danych na kategorie, a następnie zakodowanie jako dane
        # numeryczne w nowym DF
        cols_objects = self.df.columns[self.df.dtypes == "object"].tolist()

        # Kodowanie danych katygorycznych z uzyciem etykiet
        for cols in cols_objects:
            self.df[cols] = self.df[cols].astype("category")
            self.df[cols] = self.df[cols].cat.codes
            self.df.loc[self.df[cols] == -1, cols] = np.nan

        # Zamiana na float dla ujednolicenia typu
        self.df = self.df.astype(np.float64)

        # Podzielenie Dataframe na zawierające NaN w wybranej kolumnie i
        # wypełnione
        df_all_nan = self.df[self.df[self.col].isnull()]
        df_no_nan = self.df[~self.df[self.col].isnull()]

        df_no_nan = df_no_nan.dropna()

        features_all_nan = df_all_nan.drop(self.col, axis=1)
        target_all_nan = df_all_nan[self.col]
        features_no_nan = df_no_nan.drop(self.col, axis=1)
        target_no_nan = df_no_nan[self.col]

        # Utworzenie listy z ID kolumn zawierających dane kategoryczne
        cat_arr = []

        for cols in features_all_nan:
            if cols in cols_objects:
                cat_arr.append(features_all_nan.columns.get_loc(cols))

        return (
            features_all_nan,
            target_all_nan,
            features_no_nan,
            target_no_nan,
            cat_arr,
        )


def naucz_model(dane):
    features = dane.features_no_nan.to_numpy()
    target = dane.target_no_nan.to_numpy()

    # Wydzielenie zbiorów uczących i testowych
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=109
    )

    # Nauka modelu
    clf = HistGradientBoostingClassifier(
        max_iter=100, categorical_features=dane.cat_arr
    ).fit(x_train, y_train)

    # Test skuteczności modelu
    y_pred = clf.predict(x_test)
    print("Skuteczność: ", metrics.accuracy_score(y_test, y_pred))

    return clf


def wypelnij_puste(clf, dane):
    x_fill = dane.features_all_nan.to_numpy()

    y_fill = clf.predict(x_fill)

    filled = np.append(x_fill, y_fill)

    print(filled)


dane = Dane()

clf = naucz_model(dane)

wypelnij_puste(clf, dane)
