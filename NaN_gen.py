from cmath import nan
from typing import Type
import numpy as np
import pandas as pd
from sympy import true
from random import sample


# Funkcje wyboru
def wybory():
    print("Aby wyjść z programu, wpisz \"koniec\".\n")

    df = wybor_pliku()
    col = wybor_kolumny(df)
    nan_rate = wybor_procentow()

    return df, col, nan_rate


# Wybranie i wczytanie pliku do pracy
def wybor_pliku():
    while true:
        file = input(
            "Wpisz nazwę pliku lub wciśnij Enter aby wybrać domyślny: "
        )
        if not file:
            print("Wybrano domyślny plik indexData_NYA.csv\n")
            df = pd.read_csv("indexData_NYA.csv")
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


# Wybranie kolumny do dziurawienia
def wybor_kolumny(df):
    cols = df.columns.to_list()
    print("Dostępne kolumny:")
    print(cols)
    while true:
        col = input("Wpisz etykietę kolumny do usunięcia wartości: ")
        if col in cols:
            print("Wybrano kolumnę " + col + "\n")
            break
        elif col == "koniec":
            exit()
        else:
            print("Nie ma takiej kolumny!\n")

    return col


# Wybranie ile % pustych wartości ma zostać stworzone
def wybor_procentow():
    while true:
        nan_rate = input("Ile % wartości ma być usuniętych?: ")
        if nan_rate == "koniec":
            exit()
        else:
            try:
                nan_rate = float(nan_rate)
                break
            except ValueError:
                print("Błąd wartości: proszę podać liczbę rzeczywistą,")
                print("z kropką jako separator.\n")

    return nan_rate


# Usuwanie wybranej ilości wartości z wskazanej kolumny i zapisanie
# jako nowy plik
def usuwanie_wartosci(df, col, nan_rate):
    ind = len(df.index)
    num_to_rem = int(nan_rate * 0.01 * ind)
    ind_to_rem = sample(range(ind), num_to_rem)
    ind_to_rem.sort()
    for i in ind_to_rem:
        df.loc[i, col] = np.nan
    df.to_csv("test.csv", index=False)

df, col, nan_rate = wybory()

usuwanie_wartosci(df, col, nan_rate)
