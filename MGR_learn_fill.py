import numpy as np
import pandas as pd
from sympy import true
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from math import sqrt

def typ_danych(dane):

    # Wybranie kolumny do wypełnienia
    col = dane.cols_to_fill[0]

    # Sprawdzenie typu danych i wywołanie odpowiedniej funkcji
    if (
        dane.df[col].dtypes == "object"
        or dane.df[col].dtypes == "category"
    ):
        dane.przygotowanie_danych_kategoryczne()
    else:
        dane.przygotowanie_danych_liczbowe()

def kategoryczne():
    pass

def liczbowe():
    pass

def naucz_i_wypełnij(dane):

    while true:
        if dane.cols_to_fill:

            dane.przygotuj_dane()

            # Wydzielenie zbiorów uczących i testowych
            x_train, x_test, y_train, y_test = train_test_split(
                dane.features_no_nan,
                dane.target_no_nan,
                test_size=0.3,
                random_state=109,
            )

            # Nauka modelu
            clf = HistGradientBoostingClassifier(
                max_iter=100, categorical_features=dane.cat_arr
            ).fit(x_train, y_train)

            # Test skuteczności modelu
            y_pred = clf.predict(x_test)
            print(
                "Skuteczność nauczania wypełniania kolumny",
                dane.col,
                ":",
                metrics.accuracy_score(y_test, y_pred),
            )

            dane.target_all_nan = clf.predict(dane.features_all_nan)

            dane.przywroc_df()
        else:
            dane.zapisz_plik()