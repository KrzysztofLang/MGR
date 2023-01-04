from cmath import nan
import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sympy import false, true
from MGR_TempFill_class import TempFiller

default = "adult_holes.csv"


class Data:
    def __init__(self) -> None:

        # Przygotowanie encoderów
        self.enc_label_features_all = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )

        self.enc_label_features_no = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )

        self.enc_label_target = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )

        self.enc_ohe_features = OneHotEncoder(
            sparse=false, handle_unknown="ignore"
        )

        # Wybranie pliku do wypełniania
        self.choose_file()

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
            self.cols_to_fill[cols] = self.df[cols].isna().sum()

        self.cols_to_fill = list(
            dict(
                sorted(self.cols_to_fill.items(), key=lambda item: item[1])
            ).keys()
        )

        # Dodanie kolumny przechowującej oryginalne ID rekordów
        self.df.insert(0, "keep_id", self.df.index.tolist())
        print(self.df)
        print(self.df.info())

    # Wybranie i wczytanie pliku do pracy
    def choose_file(self):
        while true:
            self.file = input(
                "Wpisz nazwę pliku lub wciśnij Enter aby wybrać domyślny: "
            )

            if not self.file:
                print("Wybrano domyślny plik " + default + "\n")
                self.file = default
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
    def prepare_categorical(self, col):

        self.columns_temp = list(self.df)
        self.columns_temp.remove(col)

        # Podzielenie Dataframe na zawierające NaN w wybranej kolumnie i
        # wypełnione
        df_all_nan = self.df[self.df[col].isnull()]
        df_no_nan = self.df[~self.df[col].isnull()]

        # Podzielenie DF na dane uczące i cele
        self.features_all_nan = df_all_nan.drop(col, axis=1)
        self.target_all_nan = df_all_nan[col]
        self.features_no_nan = df_no_nan.drop(col, axis=1)
        self.target_no_nan = df_no_nan[col]

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
        self.features_no_nan = self.enc_label_features_no.fit_transform(
            self.features_no_nan
        )

        self.target_no_nan = self.enc_label_target.fit_transform(
            self.target_no_nan.reshape(-1, 1)
        ).ravel()

        self.features_all_nan = self.enc_label_features_all.fit_transform(
            self.features_all_nan
        )

        # Usunięcie nazwy wypełnianej kolumny z listy
        del self.cols_to_fill[0]

    # Przygotowanie danych do dalszej pracy w wypadku gdy
    # wybrana kolumna zawiera dane liczbowe
    def prepare_numerical(self, col):

        # Lista nazw kolumn z danymi uczącymi
        self.columns_temp = list(self.df)
        self.columns_temp.remove(col)

        # Lista ID rekordów z pustymi miejscami w wybranej kolumnie
        # oraz wypełnionych
        full_rows = self.df[~self.df[col].isna()]
        full_rows.reset_index(drop=True, inplace=True)
        last_full_id = full_rows.tail(1).index.tolist()
        last_full_id = last_full_id[0]
        nan_rows = self.df[self.df[col].isna()]
        nan_rows.reset_index(drop=True, inplace=True)

        self.df = pd.concat([full_rows, nan_rows])

        del full_rows
        del nan_rows

        # Podział na dane uczące i cel
        features = self.df.drop(col, axis=1)
        target = self.df[col]
                                 
        self.temp_filler = TempFiller()

        for column in features.iteritems():
            filled = self.temp_filler.temp_fill(column)
            features[column[0]] = filled

        # Konwersja DF na numpy array
        features = features.to_numpy()
        target = target.to_numpy()

        # Kodowanie One Hot Encoding
        features = self.enc_ohe_features.fit_transform(features)

        print(np.shape(features)[1])


        # Podzielenie tablic na zawierające NaN w wybranej kolumnie i
        # wypełnione
        self.features_no_nan = features[: last_full_id + 1, :]
        self.features_all_nan = features[last_full_id + 1 :, :]

        self.target_no_nan = target[: last_full_id + 1]
        self.target_all_nan = target[last_full_id + 1 :]
        print("Po usuwaniu: ", datetime.datetime.now())
        # Usunięcie nazwy wypełnianej kolumny z listy
        del self.cols_to_fill[0]

    # Przywrócenie danym ich pierwotnej formy
    def revert_categorical(self, col):

        # Przywrócenie danym kategorycznym odpowiednich wartości
        self.features_no_nan = self.enc_label_features_no.inverse_transform(
            self.features_no_nan
        )
        self.target_no_nan = self.enc_label_target.inverse_transform(
            self.target_no_nan.reshape(-1, 1)
        ).ravel()
        self.features_all_nan = self.enc_label_features_all.inverse_transform(
            self.features_all_nan
        )
        self.target_all_nan = self.enc_label_target.inverse_transform(
            self.target_all_nan.reshape(-1, 1)
        ).ravel()

        # Zamiana Numpy Array na DataFrame
        self.features_all_nan = pd.DataFrame(
            self.features_all_nan, columns=self.columns_temp
        )
        self.target_all_nan = pd.Series(self.target_all_nan, name=col)
        self.features_no_nan = pd.DataFrame(
            self.features_no_nan, columns=self.columns_temp
        )
        self.target_no_nan = pd.Series(self.target_no_nan, name=col)

        self.df_all_nan = self.features_all_nan.join(self.target_all_nan)
        self.df_no_nan = self.features_no_nan.join(self.target_no_nan)

        self.df = pd.concat([self.df_all_nan, self.df_no_nan])

        self.df = self.df.convert_dtypes(convert_string=False)

    def revert_numerical(self, col):

        # Przywrócenie danym kategorycznym odpowiednich wartości
        self.features_no_nan = self.enc_ohe_features.inverse_transform(
            self.features_no_nan
        )

        self.features_all_nan = self.enc_ohe_features.inverse_transform(
            self.features_all_nan
        )
        # Zamiana Numpy Array na DataFrame
        self.features_all_nan = pd.DataFrame(
            self.features_all_nan, columns=self.columns_temp
        )

        self.target_all_nan = pd.Series(self.target_all_nan, name=col)

        self.features_no_nan = pd.DataFrame(
            self.features_no_nan, columns=self.columns_temp
        )

        self.target_no_nan = pd.Series(self.target_no_nan, name=col)

        self.df_all_nan = self.features_all_nan.join(self.target_all_nan)
        self.df_no_nan = self.features_no_nan.join(self.target_no_nan)

        self.df = pd.concat([self.df_all_nan, self.df_no_nan])
        self.df = self.df.convert_dtypes(convert_string=False)

    def save_file(self):
        self.df = self.df[self.columns]
        self.df.to_csv("filled_" + self.file, index=False)
        exit()
