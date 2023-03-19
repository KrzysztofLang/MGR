import glob
import pandas as pd
from easygui import *


class AccuracyTester:
    def __init__(self) -> None:
        self.load_data()
        self.test()

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

    def test(self):

        accuracy = pd.DataFrame(columns=["column","correct"])
        
        chunk = len(self.journal)
        chunk = 100 / chunk
        progress = 0
        last_prog = 0
        print("Postęp: ", round(progress), "%")
        for item in self.journal.itertuples(index=False):
            orig_item = self.original.loc[item[0], item[1]]
            fill_item = self.filled.loc[item[0], item[1]]
            if orig_item == fill_item:
                if item[1] in accuracy["column"].unique():
                    index = accuracy.index[accuracy["column"] == item[1]].values[0]
                    accuracy.loc[accuracy.index[accuracy["column"] == item[1]]] = [item[1], accuracy.at[index, "correct"] + 1]
                else:
                    accuracy.loc[len(accuracy)] = [item[1], 1]
            if round(progress) % 10 == 0 and round(progress) != last_prog:
                    print("Postęp: ", round(progress), "%")
                    last_prog = round(progress)
            progress += chunk
                

        
        print(accuracy)
        
