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
from math import sqrt


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
    print("Aby wyjść z programu, wpisz \"koniec\".\n")

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
        col = input("Etykieta kolumny do wypełnienia: ")
        if col in cols:
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

    for cols in cols_objects:
        df[cols] = df[cols].cat.codes
        df.loc[df[cols] == -1, cols] = np.nan

    # Zamiana na float dla ujednolicenia typu
    df = df.astype(np.float64)

    # Podzielenie Dataframe na zawierające NaN w wybranej kolumnie i
    # wypełnione
    df_all_nan = df[df[col].isnull()]
    df_no_nan = df[~df[col].isnull()]

    print(df_all_nan.info())
    print(df_no_nan.info())
    
    return df_all_nan, df_no_nan


"""     # Wydzielenie danych uczących
    df_no_nan_target = df_no_nan.pop(col)
    df_all_nan_target = df_all_nan.pop(col)

    return df_all_nan, df_no_nan, df_all_nan_target, df_no_nan_target """


# Wczytanie pliku danych
df, col = wybory()

df_all_nan, df_no_nan = przygotowanie_danych(df, col)

target_column = [col] 
predictors = list(set(list(df_no_nan.columns)) - set(target_column))
df_no_nan[predictors] = df_no_nan[predictors] / df_no_nan[predictors].max()
print(df_no_nan.describe())

X = df_no_nan[predictors].values
y = df_no_nan[target_column].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=40
)
print(X_train.shape)
print(X_test.shape)

print(y_train)

# y_train = to_categorical(y_train)
#  
# y_test = to_categorical(y_test)

count_classes = y_test.shape[1]
print(count_classes)

model = Sequential()
model.add(Dense(500, activation="relu", input_dim=7))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))
model.add(Dense(count_classes, activation="softmax"))

# Compile the model
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=20)

""" (
    df_all_nan,
    df_no_nan,
    df_all_nan_target,
    df_no_nan_target,
) = przygotowanie_danych()

model = get_basic_model()
model.fit(df_no_nan, df_no_nan_target, epochs=15, batch_size=2) """
