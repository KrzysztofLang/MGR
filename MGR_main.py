import numpy as np
import pandas as pd
from sympy import true
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from math import sqrt
import re


class Dane:
    def __init__(self) -> None:

        self.enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )
        # Wybranie pliku do wypełniania
        print('Aby wyjść z programu, wpisz "koniec".\n')
        self.df = self.wybor_pliku()
        self.cols_to_fill = {}
        for cols in self.df.columns[self.df.isna().any()].tolist():
            if (
                self.df[cols].dtypes == "object"
                or self.df[cols].dtypes == "category"
            ):
                self.cols_to_fill[cols] = self.df[cols].isna().sum()

        self.cols_to_fill = list(
            dict(
                sorted(self.cols_to_fill.items(), key=lambda item: item[1])
            ).keys()
        )
        print(self.cols_to_fill)
        exit()

    # Wybranie i wczytanie pliku do pracy
    @staticmethod
    def wybor_pliku():
        while true:
            file = input(
                "Wpisz nazwę pliku lub wciśnij Enter aby wybrać domyślny: "
            )

            if not file:
                print("Wybrano domyślny plik adult_holes.csv\n")
                df = pd.read_csv("adult_holes.csv")
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

    # Przygotowanie danych do dalszej pracy w wypadku gdy
    # Wybrana kolumna zawiera dane kategoryczne
    def przygotowanie_danych_kategoryczne(self):
        # Zamiana typow danych na kategorie, a następnie zakodowanie jako dane
        # numeryczne w nowym DF
        cols_with_objects = self.df.columns[
            self.df.dtypes == "object"
        ].tolist()

        # Kodowanie danych katygorycznych z uzyciem etykiet

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
            if cols in cols_with_objects:
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
