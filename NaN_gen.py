from cmath import nan
from typing import Type
import numpy as np
import pandas as pd
from sympy import true
from random import sample

def wybory():   ##Uruchamianie funkcji wyboru
    print("Aby wyjśćz programu, wpisz \"koniec\".")
    col = wybor_kolumny()
    nanRate = wybor_procentow()

    return col, nanRate

def wybor_kolumny():    ##Wybranie kolumny do dziurawienia
    cols = df.columns.to_list()
    print("Dostępne kolumny:")
    print(cols)
    while true:
        col = input("Etykieta kolumny usunięcia wartości: ")
        if col in cols:
            print('Wybrano poprawną kolumnę ' + col)
            break
        elif col == 'koniec':
            exit()
        else:
            print('Nie ma takiej kolumny!')

    return col

def wybor_procentow():    ##Wybranie ile % pustych wartości ma zostać stworzone
    while true:
        nanRate = input("Ile % wartości ma być usuniętych?: ")
        if nanRate == 'koniec':
            exit()
        else:
            try:
                nanRate = float(nanRate)
                break
            except ValueError:
                print("Błąd wartości: proszę podać liczbę rzeczywistą, z kropką jako separator.")

    return nanRate

def usuwanie_wartosci(df, col, nanRate):    ##Usuwanie wybranej ilości wartości z wskazanej kolumny i zapisanie jako nowy plik
    ind = len(df.index)
    numToRem = int(nanRate*0.01*ind)
    indToRem = sample(range(ind),numToRem)
    indToRem.sort()
    for i in indToRem:
        df.loc[i,col] = np.nan
    df.to_csv('test.csv', index=False)

##Wczytanie pliku danych
df = pd.read_csv("indexData_NYA.csv")

print(df.info())

col, nanRate = wybory()

usuwanie_wartosci(df, col, nanRate)