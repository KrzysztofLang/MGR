import glob
import random
import io

import numpy as np
import pandas as pd
import os.path
from random import sample
from easygui import *


class NanGen:
    def __init__(self) -> None:
        df, cols, file = self.choices()
        self.generate_nan(df, cols, file)

    @staticmethod
    def choices():
        # Wybranie i wczytanie pliku do pracy
        files = glob.glob("./*.csv")
        files = [x for x in files if "holes" not in x]

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

        if file:
            df = pd.read_csv(file)
        else:
            exit()

        # Wybranie kolumny do dziurawienia
        all_cols = df.columns.to_list()

        cols = multchoicebox(
            "Wybierz kolumny do dziurawienia:", "NaN Generator", all_cols
        )

        if cols:
            pass
        else:
            exit()

        return df, cols, file

    # Usuwanie wartości z wskazanych kolumn i zapisanie
    # jako nowy plik
    @staticmethod
    def generate_nan(df, cols, file):
        journal = pd.DataFrame(columns=["row", "column"])

        for col in cols:
            ind = len(df.index)
            num_to_rem = int(random.randrange(5, 15) * 0.01 * ind)
            ind_to_rem = sample(range(ind), num_to_rem)
            ind_to_rem.sort()
            for i in ind_to_rem:
                df.loc[i, col] = np.nan
                journal.loc[len(journal)] = [i, col]

        # Zapisanie do pliku z nową nazwą
        i = 1
        while True:
            name = file[2 : len(file) - 4] + "_holes_" + str(i) + ".csv"
            if i > 9:
                msgbox("Za dużo plików!", "NaN Generator")
                exit()
            elif os.path.isfile(name):
                i += 1
            else:
                break

        # Przywrócenie kolumnom właściwych typów
        df = df.convert_dtypes(convert_string=False)

        # Wyświetlenie okna z informacjami o utworzonym pliku
        buffer = io.StringIO()
        df.info(buf=buffer)
        info = buffer.getvalue()
        if codebox(
            "Wynik zostanie zapisany do pliku "
            + name
            + ".\nInformacje o danych po dziurawieniu:",
            "NaN Generator",
            info,
        ):
            df.to_csv(name, index=False)
            journal.to_csv(name[: len(name) - 4] + "_journal.csv", index=False)
        else:
            exit()
