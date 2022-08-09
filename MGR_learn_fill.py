from sympy import true
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from math import sqrt
import statsmodels.api as sm
import datetime


def typ_danych(dane):
    # Wybranie kolumny do wypełnienia
    col = dane.cols_to_fill[0]
    print(col)
    # Sprawdzenie typu danych i wywołanie odpowiedniej funkcji
    if dane.df[col].dtypes == "object" or dane.df[col].dtypes == "category":
        type = "cat"
    else:
        type = "num"
    
    print(type)
    return type, col


def kategoryczne(dane, col):
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
        col,
        ":",
        metrics.accuracy_score(y_test, y_pred),
    )

    dane.target_all_nan = clf.predict(dane.features_all_nan)

    dane.przywroc_df_kategoryczne(col)


def liczbowe(dane, col):
    # Wydzielenie zbiorów uczących i testowych
    print("Wydzielanie danych: ", datetime.datetime.now())
    x_train, x_test, y_train, y_test = train_test_split(
        dane.features_no_nan,
        dane.target_no_nan,
        test_size=0.3,
        random_state=109,
    )

    # Nauka modelu
    print("Nauka modelu: ", datetime.datetime.now())
    model = sm.OLS(y_train, x_train, missing="drop")
    print("Fit: ", datetime.datetime.now())
    result = model.fit()

    # Test skuteczności modelu
    print("Accuracy train {:.3f}".format(model.score(x_train, y_train)))
    print("Accuracy test {:.3f}".format(model.score(x_test, y_test)))

    dane.target_all_nan = model.predict(dane.features_all_nan)

    dane.przywroc_df_liczbowe(col)

# Główna funkcja, wywoływana z głównego pliku
def naucz_i_wypełnij(dane):
    while true:
        print(dane.df.info())
        if dane.cols_to_fill:
            type, col = typ_danych(dane)
            match type:
                case "cat":
                    dane.przygotowanie_danych_kategoryczne(col)
                    kategoryczne(dane, col)
                case "num":
                    dane.przygotowanie_danych_liczbowe(col)
                    liczbowe(dane, col)

        else:
            dane.zapisz_plik()
