import glob
import random

import numpy as np
import pandas as pd
from random import sample
from easygui import *


def choices():
    # Wybranie i wczytanie pliku do pracy
    files = glob.glob("./*.csv")

    file = choicebox(
        " _   _       _   _    _____\n"
        + "| \ | |     | \ | |  / ____|\n"
        + "|  \| | __ _|  \| | | |  __  ___ _ __\n"
        + "| . ` |/ _` | . ` | | | |_ |/ _ \ '_ \\\n"
        + "| |\  | (_| | |\  | | |__| |  __/ | | |\n"
        + "|_| \_|\__,_|_| \_|  \_____|\___|_| |_|\n\n"
        + "Wybierz plik do dziurawienia:",
        "NaN Generator",
        files,
    )

    df = pd.read_csv(file)

    # Wybranie kolumny do dziurawienia
    all_cols = df.columns.to_list()

    cols = multchoicebox(
        "Wybierz kolumny do dziurawienia:", "NaN Generator", all_cols
    )

    return df, cols, file


# Usuwanie wybranej ilości wartości z wskazanych kolumn i zapisanie
# jako nowy plik
def generate_nan(df, cols, file):
    for col in cols:
        ind = len(df.index)
        num_to_rem = int(random.randrange(5, 15) * 0.01 * ind)
        ind_to_rem = sample(range(ind), num_to_rem)
        ind_to_rem.sort()
        for i in ind_to_rem:
            df.loc[i, col] = np.nan
        df.to_csv("holes_" + file[2:], index=False)

    print(df.info())


df, cols, file = choices()

generate_nan(df, cols, file)
