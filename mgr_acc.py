import glob
import pandas as pd
from easygui import *


class AccuracyTester:
    def __init__(self) -> None:
        self.load_data()

    def load_data(self):
        # Wybranie i wczytanie pliku do pracy
        all_files = glob.glob("./*.csv")
        files = [x for x in all_files if "filled" in x]

        file = choicebox(
            "    /\             |__   __|      | |\n"
            + "   /  \   ___ ___     | | ___  ___| |_ ___ _ __ \n"
            + "  / /\ \ / __/ __|    | |/ _ \/ __| __/ _ \ '__|\n"
            + " / ____ \ (_| (__     | |  __/\__ \ ||  __/ |\n"
            + "/_/    \_\___\___|    |_|\___||___/\__\___|_|\n\n"
            + "Który plik chcesz zweryfikować:",
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