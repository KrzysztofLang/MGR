import glob
import pandas as pd
from easygui import *


class AccuracyTester:
    def __init__(self) -> None:
        self.load_data()

    def load_data(self):
        # Wybranie i wczytanie pliku do pracy
        logo = (
            "    ___                ______          __\n"
            + "   /   | __________   /_  __/__  _____/ /____  _____\n"
            + "  / /| |/ ___/ ___/    / / / _ \/ ___/ __/ _ \/ ___/\n"
            + " / ___ / /__/ /__     / / /  __(__  ) /_/  __/ /\n"
            + "/_/  |_\___/\___/    /_/  \___/____/\__/\___/_/\n\n"
        )

        all_files = glob.glob("./*.csv")
        files = [x for x in all_files if "filled" in x]

        if len(files) == 0:
            msgbox(
                logo
                + "Nie znaleziono odpowiednich plików.\n"
                + "Upewnij się, że w folderze w którym uruchamiasz program"
                + " znajdują się dostosowane pliki.",
                "Accuracy Test",
            )
            exit()
        elif len(files) == 1:
            if ccbox(
                logo + "Znaleziono tylko 1 plik: " + files[0], "Accuracy Test"
            ):
                file = files[0]
            else:
                exit()
        else:
            file = choicebox(
                logo + "Wybierz plik do weryfikacji:",
                "Accuracy Test",
                files,
            )

        if file:
            self.filled = pd.read_csv(file)
            self.original = pd.read_csv(file.split("_holes_")[0] + ".csv")
            self.journal = pd.read_csv(file.split("filled")[0] + "journal.csv")
        else:
            exit()

        self.filled.info()
        self.original.info()
        self.journal.info()

    def test(self):
        pass
