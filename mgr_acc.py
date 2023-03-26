import glob
import pandas as pd
from easygui import *


class AccuracyTester:
    def __init__(self) -> None:
        self.load_data()
        self.test_accuracy()
        self.test_deviation()

        if msgbox(
            "Wyniki zostaną zapisane do plików:\n"
            + self.name
            + "_acc.csv\n"
            + self.name
            + "_aad.csv",
            "Test skuteczności wypełnienia",
        ):
            self.acc.to_csv(self.name + "_acc.csv", index=False)
            self.aad.to_csv(self.name + "_aad.csv", index=False)
        else:
            exit()

    def load_data(self):
        # Wybranie i wczytanie pliku do pracy
        logo = (
            "    ___                ______          __\n"
            + "   /   | __________   /_  __/__  _____/ /____  _____\n"
            + "  / /| |/ ___/ ___/    / / / _ \/ ___/ __/ _ \/ ___/\n"
            + " / ___ / /__/ /__     / / /  __(__  ) /_/  __/ /\n"
            + "/_/  |_\___/\___/    /_/  \___/____/\__/\___/_/\n\n"
        )

        # Zczytanie odpowiednich plików w lokalizacji
        all_files = glob.glob("./*.csv")
        files = [
            x
            for x in all_files
            if "data" in x
            and "filled" in x
            and "acc" not in x
            and "aad" not in x
        ]

        # Okno wyboru pliku
        if len(files) == 0:
            msgbox(
                logo
                + "Nie znaleziono odpowiednich plików.\n"
                + "Upewnij się, że w folderze w którym uruchamiasz program"
                + " znajdują się dostosowane pliki.",
                "Test skuteczności wypełnienia",
            )
            exit()
        elif len(files) == 1:
            if ccbox(
                logo + "Znaleziono tylko 1 plik: " + files[0],
                "Test skuteczności wypełnienia",
            ):
                file = files[0]
            else:
                exit()
        else:
            file = choicebox(
                logo + "Wybierz plik do weryfikacji:",
                "Test skuteczności wypełnienia",
                files,
            )

        if file:
            # DF po wypełnieniu pustych miejsc
            self.filled = pd.read_csv(file)
            # DF oryginalny
            self.original = pd.read_csv(file.split("_holes_")[0] + ".csv")
            # DF z informacjami które elementy były wypełniane
            self.journal = pd.read_csv(file.split("filled")[0] + "journal.csv")
            # Przygotowanie sufixu do nazwy plików
            self.name = file[2 : len(file) - 4]
        else:
            exit()

    def test_accuracy(self):
        print("Obliczanie skuteczności wypełniania:")
        # DF przechowujący informacje o skuteczności wypełniania
        self.acc = pd.DataFrame(columns=["column", "correct", "count"])

        # Zmienne do wyświetlania postępu
        chunk = len(self.journal)
        chunk = 100 / chunk
        progress = 0
        last_prog = 0

        print("Postęp: ", round(progress), "%")
        # Pętla zliczająca poprawnie wypełnione elementy
        for item in self.journal.itertuples(index=False):
            # Zapisanie do zmiennych porównywanych elementów
            orig_item = self.original.loc[item[0], item[1]]
            fill_item = self.filled.loc[item[0], item[1]]

            if item[1] in self.acc["column"].unique():
                # Jeśli elementy są równe, zwiększ licznik
                # poprawnych i wszystkich sprawdzanych, jeśli nie
                # to tylko wszystkich
                index = self.acc.index[self.acc["column"] == item[1]].values[0]
                if orig_item == fill_item:
                    self.acc.loc[
                        self.acc.index[self.acc["column"] == item[1]]
                    ] = [
                        item[1],
                        self.acc.at[index, "correct"] + 1,
                        self.acc.at[index, "count"] + 1,
                    ]
                else:
                    self.acc.loc[
                        self.acc.index[self.acc["column"] == item[1]]
                    ] = [
                        item[1],
                        self.acc.at[index, "correct"],
                        self.acc.at[index, "count"] + 1,
                    ]
            else:
                self.acc.loc[len(self.acc)] = [item[1], 1, 1]

            if round(progress) % 10 == 0 and round(progress) != last_prog:
                print("Postęp: ", round(progress), "%")
                last_prog = round(progress)
            progress += chunk

        # Wypełnienie kolumny procentową wartością dokłądności wypełniania
        self.acc["acc"] = 100 * self.acc["correct"] / self.acc["count"]

    def test_deviation(self):
        print("Obliczanie średniego odchylenia:")
        # DF przechowujący informacje o średnim odchyleniu
        self.aad = pd.DataFrame(columns=["column", "dev_sum", "count"])

        # Zmienne do wyświetlania postępu
        chunk = len(self.journal)
        chunk = 100 / chunk
        progress = 0
        last_prog = 0

        print("Postęp: ", round(progress), "%")
        # Pętla sumująca odchylenie elementów
        for item in self.journal.itertuples(index=False):
            orig_item = self.original.loc[item[0], item[1]]
            if isinstance(orig_item, str):
                pass
            else:
                fill_item = self.filled.loc[item[0], item[1]]
                dev = abs(orig_item - fill_item)

                if item[1] in self.aad["column"].unique():
                    index = self.aad.index[
                        self.aad["column"] == item[1]
                    ].values[0]
                    self.aad.loc[
                        self.aad.index[self.aad["column"] == item[1]]
                    ] = [
                        item[1],
                        self.aad.at[index, "dev_sum"] + dev,
                        self.aad.at[index, "count"] + 1,
                    ]
                else:
                    self.aad.loc[len(self.aad)] = [item[1], dev, 1]

            if round(progress) % 10 == 0 and round(progress) != last_prog:
                print("Postęp: ", round(progress), "%")
                last_prog = round(progress)
            progress += chunk

        # Wypełnienie kolumny procentową wartością dokładności wypełniania
        self.aad["aad"] = self.aad["dev_sum"] / self.aad["count"]
        self.aad.drop(["dev_sum", "count"], axis=1, inplace=True)
