from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sympy import true
from MGR_Data_class import PrepareOld
import numpy as np


def check_datatype(data):
    # Wybranie kolumny do wypełnienia
    col = data.cols_to_fill[0]

    # Sprawdzenie typu danych
    if data.df[col].dtypes == "object" or data.df[col].dtypes == "category":
        type = "cat"
    else:
        type = "num"

    return type, col


def fill_categorical(data, col):
    print("Wypełniana kolumna: ", col)
    # Wydzielenie zbiorów uczących i testowych
    x_train, x_test, y_train, y_test = train_test_split(
        data.features_no_nan,
        data.target_no_nan,
        test_size=0.3,
        random_state=109,
    )
    np.savetxt("x_train.csv", x_train, delimiter=",")
    # Nauka modelu
    clf = HistGradientBoostingClassifier(
        max_iter=100, categorical_features=data.cat_arr
    ).fit(x_train, y_train)

    # Test skuteczności modelu
    y_pred = clf.predict(x_test)
    print(
        "Skuteczność nauczania wypełniania: ",
        metrics.accuracy_score(y_test, y_pred),
    )

    data.target_all_nan = clf.predict(data.features_all_nan)


def fill_numerical(data, col):
    # Wydzielenie zbiorów uczących i testowych
    print("Wypełniana kolumna: ", col)
    x_train, x_test, y_train, y_test = train_test_split(
        data.features_no_nan[:, 1:],
        data.target_no_nan,
        test_size=0.3,
        random_state=0,
    )

    # Nauka modelu
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Test skuteczności modelu
    print("Accuracy train {:.3f}".format(model.score(x_train, y_train)))
    print("Accuracy test {:.3f}".format(model.score(x_test, y_test)))
    data.target_all_nan = model.predict(data.features_all_nan[:, 1:])


# Główna funkcja, wywoływana z głównego pliku
def fill_nan(data):
    print("fill_nan start")
    match data.algorithm:
        case "Stary":
            while true:
                if data.cols_to_fill:
                    type, col = check_datatype(data)
                    match type:
                        case "num":
                            PrepareOld.prepare_numerical(data, col)
                            fill_numerical(data, col)
                            PrepareOld.revert_numerical(data, col)
                        case "cat":
                            PrepareOld.prepare_categorical(data, col)
                            fill_categorical(data, col)
                            PrepareOld.revert_categorical(data, col)
                else:
                    data.df = data.df[data.columns]
                    print(data.df.info())
                    data.df.sort_values("keep_id", inplace=True)
                    data.df.drop("keep_id", axis=1, inplace=True)
                    data.df.to_csv("filled_" + data.file[2:], index=False)
                    exit()
        case "Nowy":
            print("Algorytm nie zaimplementowany")
            exit()
