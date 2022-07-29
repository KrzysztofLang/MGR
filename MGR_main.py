import numpy as np
import pandas as pd
from sympy import true
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder
from math import sqrt


class Dane:
    def __init__(self) -> None:

        # Przygotowanie encoderów
        self.enc_features_all = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )

        self.enc_features_no = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )

        self.enc_target = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )

        # Wybranie pliku do wypełniania
        print('Aby wyjść z programu, wpisz "koniec".\n')
        self.wybor_pliku()

        # Lista wszystkichh kolumn
        self.columns = list(self.df)

        # Lista kolumn z damnymi kategorycznymi
        self.cols_with_objects = self.df.columns[
            self.df.dtypes == "object"
        ].tolist()

        # Utworzenie listy kolumn do wypełnienia,
        # posortowana od najmniejszej do największej ilości NaN
        self.cols_to_fill = {}
        for cols in self.df.columns[self.df.isna().any()].tolist():
            if (
                self.df[cols].dtypes == "object"
                or self.df[cols].dtypes == "category"
            ):
                self.cols_to_fill[cols] = self.df[cols].isna().sum()

        self.cols_to_fill = list(
            dict(
                sorted(self.cols_to_fill.items(), key=lambda item: item[1])
            ).keys()
        )

    # Wybranie i wczytanie pliku do pracy
    def wybor_pliku(self):
        while true:
            self.file = input(
                "Wpisz nazwę pliku lub wciśnij Enter aby wybrać domyślny: "
            )

            if not self.file:
                print("Wybrano domyślny plik adult_holes.csv\n")
                self.file = "adult_holes.csv"
                break
            elif self.file == "koniec":
                exit()
            else:
                try:
                    print("Wybrano plik " + self.file + "\n")
                    break
                except Exception:
                    print(
                        "Wpisano niepoprawną nazwę pliku, proszę upewnić się"
                    )
                    print("czy plik znajduje się w folderze programu.\n")

        self.df = pd.read_csv(self.file)

    # Przygotowanie danych do dalszej pracy w wypadku gdy
    # wybrana kolumna zawiera dane kategoryczne
    def przygotowanie_danych_kategoryczne(self):

        # Wybranie kolumny do wypełnienia
        self.col = self.cols_to_fill[0]

        # Podzielenie Dataframe na zawierające NaN w wybranej kolumnie i
        # wypełnione
        df_all_nan = self.df[self.df[self.col].isnull()]
        df_no_nan = self.df[~self.df[self.col].isnull()]

        # Podzielenie DF na dane uczące i cele
        self.features_all_nan = df_all_nan.drop(self.col, axis=1)
        self.target_all_nan = df_all_nan[self.col]
        self.features_no_nan = df_no_nan.drop(self.col, axis=1)
        self.target_no_nan = df_no_nan[self.col]

        # Utworzenie list z nazwami kolumn
        self.features_all_nan_columns = list(self.features_all_nan)
        self.target_all_nan_columns = self.target_all_nan.name
        self.features_no_nan_columns = list(self.features_no_nan)
        self.target_no_nan_columns = self.target_no_nan.name

        # Utworzenie listy z ID kolumn zawierających dane kategoryczne
        self.cat_arr = []

        for cols in self.features_all_nan:
            if cols in self.cols_with_objects:
                self.cat_arr.append(
                    self.features_all_nan.columns.get_loc(cols)
                )

        # Konwersja DF na numpy array
        self.features_all_nan = self.features_all_nan.to_numpy()
        self.target_all_nan = self.target_all_nan.to_numpy()
        self.features_no_nan = self.features_no_nan.to_numpy()
        self.target_no_nan = self.target_no_nan.to_numpy()

        # Kodowanie Label Encoding
        self.features_no_nan = self.enc_features_no.fit_transform(
            self.features_no_nan
        )

        self.target_no_nan = self.enc_target.fit_transform(
            self.target_no_nan.reshape(-1, 1)
        ).ravel()

        self.features_all_nan = self.enc_features_all.fit_transform(
            self.features_all_nan
        )

        del self.cols_to_fill[0]

    # Przywrócenie danym ich pierwotnej formy
    def przywroc_df(self):

        # Przywrócenie danym kategorycznym odpowiednich wartości
        self.features_no_nan = self.enc_features_no.inverse_transform(
            self.features_no_nan
        )
        self.target_no_nan = self.enc_target.inverse_transform(
            self.target_no_nan.reshape(-1, 1)
        ).ravel()
        self.features_all_nan = self.enc_features_all.inverse_transform(
            self.features_all_nan
        )
        self.target_all_nan = self.enc_target.inverse_transform(
            self.target_all_nan.reshape(-1, 1)
        ).ravel()

        # Zamiana Numpy Array na DataFrame
        self.features_all_nan = pd.DataFrame(
            self.features_all_nan, columns=self.features_all_nan_columns
        )
        self.target_all_nan = pd.Series(
            self.target_all_nan, name=self.target_all_nan_columns
        )
        self.features_no_nan = pd.DataFrame(
            self.features_no_nan, columns=self.features_no_nan_columns
        )
        self.target_no_nan = pd.Series(
            self.target_no_nan, name=self.target_no_nan_columns
        )

        self.df_all_nan = self.features_all_nan.join(self.target_all_nan)
        self.df_no_nan = self.features_no_nan.join(self.target_no_nan)

        self.df = pd.concat([self.df_all_nan, self.df_no_nan])

        print(self.df.info())

    def zapisz_plik(self):
        new_file = "filled_" + self.file
        self.df.to_csv(new_file, index=False)
        exit()


def naucz_i_wypełnij(dane):

    while true:
        if dane.cols_to_fill:

            dane.przygotowanie_danych_kategoryczne()

            # Wydzielenie zbiorów uczących i testowych
            x_train, x_test, y_train, y_test = train_test_split(
                dane.features_no_nan,
                dane.target_no_nan,
                test_size=0.3,
                random_state=109,
            )

            # Nauka modelu
            clf = HistGradientBoostingClassifier(
                max_iter=100, categorical_features=dane.cat_arr
            ).fit(x_train, y_train)

            # Test skuteczności modelu
            y_pred = clf.predict(x_test)
            print(
                "Skuteczność nauczania wypełniania kolumny",
                dane.col,
                ":",
                metrics.accuracy_score(y_test, y_pred),
            )

            dane.target_all_nan = clf.predict(dane.features_all_nan)

            dane.przywroc_df()
        else:
            dane.zapisz_plik()


dane = Dane()

naucz_i_wypełnij(dane)
