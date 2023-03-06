import datetime

from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sympy import true


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
    # Wydzielenie zbiorów uczących i testowych
    x_train, x_test, y_train, y_test = train_test_split(
        data.features_no_nan,
        data.target_no_nan,
        test_size=0.3,
        random_state=109,
    )

    # Nauka modelu
    clf = HistGradientBoostingClassifier(
        max_iter=100, categorical_features=data.cat_arr
    ).fit(x_train, y_train)

    # Test skuteczności modelu
    y_pred = clf.predict(x_test)
    print(
        "Skuteczność nauczania wypełniania kolumny",
        col,
        ":",
        metrics.accuracy_score(y_test, y_pred),
    )

    data.target_all_nan = clf.predict(data.features_all_nan)

    data.revert_categorical(col)


def fill_numerical(data, col):
    # Wydzielenie zbiorów uczących i testowych
    print("Wydzielanie danych: ", datetime.datetime.now())

    x_train, x_test, y_train, y_test = train_test_split(
        data.features_no_nan[:, 1:],
        data.target_no_nan,
        test_size=0.3,
        random_state=0,
    )

    # Nauka modelu
    model = LinearRegression()
    print("Fit: ", datetime.datetime.now())
    model.fit(x_train, y_train)

    # Test skuteczności modelu
    print("Accuracy train {:.3f}".format(model.score(x_train, y_train)))
    print("Accuracy test {:.3f}".format(model.score(x_test, y_test)))
    print("Predict: ", datetime.datetime.now())
    data.target_all_nan = model.predict(data.features_all_nan[:, 1:])
    print("Koniec predict: ", datetime.datetime.now())
    data.revert_numerical(col)


# Główna funkcja, wywoływana z głównego pliku
def fill_nan(data):
    while true:
        if data.cols_to_fill:
            type, col = check_datatype(data)
            match type:
                case "num":
                    data.prepare_numerical(col) 
                    fill_numerical(data, col)
                case "cat":
                    data.prepare_categorical(col)
                    fill_categorical(data, col)


        else:
            data.save_file()
