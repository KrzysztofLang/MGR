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
        logo = (
            "    _   __      _   __   ______\n"
            + "   / | / /___ _/ | / /  / ____/__  ____\n"
            + "  /  |/ / __ `/  |/ /  / / __/ _ \/ __ \\\n"
            + " / /|  / /_/ / /|  /  / /_/ /  __/ / / /\n"
            + "/_/ |_/\__,_/_/ |_/   \____/\___/_/ /_/\n\n"
        )

        # Wybranie i wczytanie pliku do pracy
        files = glob.glob("./*.csv")
        files = [x for x in files if "data" in x and "holes" not in x]

        if len(files) == 0:
            msgbox(
                logo
                + "Nie znaleziono odpowiednich plików.\n"
                + "Upewnij się, że w folderze w którym uruchamiasz program"
                + " znajdują się dostosowane pliki.",
                "Przygotowanie danych",
            )
            exit()
        elif len(files) == 1:
            if ccbox(
                logo + "Znaleziono tylko 1 plik: " + files[0],
                "Przygotowanie danych",
            ):
                file = files[0]
            else:
                exit()
        else:
            file = choicebox(
                logo + "Wybierz plik do przygotowania:",
                "Przygotowanie danych",
                files,
            )

        if file:
            df = pd.read_csv(file)
        else:
            exit()

        # Wybranie kolumny do dziurawienia
        all_cols = df.columns.to_list()

        cols = multchoicebox(
            "Wybrano plik "
            + file
            + "\nWybierz kolumny do usunięcia wartości:",
            "Przygotowanie danych",
            all_cols,
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
            print("Przygotowywanie kolumny ", col, ".")
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
                msgbox("Za dużo plików!", "Przygotowanie danych")
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
            + ".\nInformacje o danych po przygotowaniu:",
            "Przygotowanie danych",
            info,
        ):
            df.to_csv(name, index=False)
            journal.to_csv(name[: len(name) - 4] + "_journal.csv", index=False)
        else:
            exit()
