from easygui import *
import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from MGR_TempFill_class import TempFiller
from MGR_learn_fill import fill_nan

default = "test_num.csv"


class Data:
    """Załadowanie danych do wypełniania"""

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
            sparse_output=False, handle_unknown="ignore"
        )

        # Wybranie pliku do wypełniania
        self.choose_file()

        # Dodanie kolumny przechowującej oryginalne ID rekordów
        self.df.insert(0, "keep_id", self.df.index.tolist())

        # Lista wszystkichh kolumn
        self.columns = list(self.df)

        # Lista kolumn z damnymi kategorycznymi
        self.cols_with_objects = self.df.columns[
            self.df.dtypes == "object"
        ].tolist()

        # Lista kolumn z damnymi typu int
        self.cols_with_int = self.df.columns[
            self.df.dtypes == "int64"
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

    # Wybranie i wczytanie pliku do pracy
    def choose_file(self):
        choices = ["Stary", "Nowy"]
        self.algorithm = choicebox(
            " _   _       _   _   ______ _ _ _\n"
            + "| \\ | |     | \\ | | |  ____(_) | |\n"
            + "|  \\| | __ _|  \\| | | |__   _| | | ___ _ __\n"
            + "| . ` |/ _` | . ` | |  __| | | | |/ _ \\ '__|\n"
            + "| |\\  | (_| | |\\  | | |    | | | |  __/ |\n"
            + "|_| \\_|\\__,_|_| \\_| |_|    |_|_|_|\\___|_|\n"
            + "Wybierz algorytm do zastosowania:",
            "NaN Filler",
            choices,
        )

        files = glob.glob("./*.csv")

        self.file = choicebox(
            "Wybierz plik do wypełnienia:", "NaN Filler", files
        )

        self.df = pd.read_csv(self.file)


class PrepareNew:
    def SumThing():
        exit()


class PrepareOld:
    """Przygotowanie danych do dalszej pracy w wypadku gdy
    wybrana kolumna zawiera dane kategoryczne
    """

    @staticmethod
    def prepare_categorical(data, col):
        # Lista nazw kolumn z danymi uczącymi
        data.column_names = list(data.df)
        data.column_names.remove(col)
        data.column_names.insert(len(data.column_names), col)

        # Podzielenie Dataframe na zawierające NaN w wybranej kolumnie i
        # wypełnione
        df_all_nan = data.df[data.df[col].isnull()]
        df_no_nan = data.df[~data.df[col].isnull()]

        # Podzielenie DF na dane uczące i cele
        data.features_all_nan = df_all_nan.drop(col, axis=1)
        data.target_all_nan = df_all_nan[col]
        data.features_no_nan = df_no_nan.drop(col, axis=1)
        data.target_no_nan = df_no_nan[col]

        # Utworzenie listy z ID kolumn zawierających dane kategoryczne
        data.cat_arr = []

        for cols in data.features_all_nan:
            if cols in data.cols_with_objects:
                data.cat_arr.append(
                    data.features_all_nan.columns.get_loc(cols)
                )

        # Konwersja DF na numpy array
        data.features_all_nan = data.features_all_nan.to_numpy()
        data.target_all_nan = data.target_all_nan.to_numpy()
        data.features_no_nan = data.features_no_nan.to_numpy()
        data.target_no_nan = data.target_no_nan.to_numpy()

        # Kodowanie Label Encoding
        data.features_no_nan = data.enc_label_features_no.fit_transform(
            data.features_no_nan
        )

        data.target_no_nan = data.enc_label_target.fit_transform(
            data.target_no_nan.reshape(-1, 1)
        ).ravel()

        data.features_all_nan = data.enc_label_features_all.fit_transform(
            data.features_all_nan
        )

        # Usunięcie nazwy wypełnianej kolumny z listy
        del data.cols_to_fill[0]

    # Przygotowanie danych do dalszej pracy w wypadku gdy
    # wybrana kolumna zawiera dane liczbowe
    @staticmethod
    def prepare_numerical(data, col):
        # Rozdzielenie danych na tabele bez i z NAN w kolumnie do wypełnienia
        full_rows = data.df[~data.df[col].isna()]
        full_rows.reset_index(drop=True, inplace=True)
        nan_rows = data.df[data.df[col].isna()]
        nan_rows.reset_index(drop=True, inplace=True)

        # Zapisanie ostatniego ID rekordów bez NAN
        last_full_id = full_rows.tail(1).index.tolist()
        last_full_id = last_full_id[0]

        # Połączenie tabel w jedną
        data.df = pd.concat([full_rows, nan_rows])

        # Usunięcie niepotrzebnych tabel
        del full_rows
        del nan_rows

        # Podział na dane uczące i cel
        features = data.df.drop(col, axis=1)
        target = data.df[col]

        # Tymczasowe wypełnienie NAN w danych uczących
        data.temp_filler = TempFiller()

        for column in features.items():
            filled = data.temp_filler.temp_fill(column)
            features[column[0]] = filled

        # Przygotowanie do kodowania wyłącznie kolumn z danymi kategorycznymi
        data.features_objects_names = features.columns[
            features.dtypes == "object"
        ].tolist()

        data.features_numbers_names = features.columns[
            features.dtypes != "object"
        ].tolist()

        # Lista nazw kolumn z danymi uczącymi
        data.column_names = (
            data.features_numbers_names + data.features_objects_names
        )
        data.column_names.insert(len(data.column_names), col)

        features_objects = features[data.features_objects_names]
        features_numbers = features[data.features_numbers_names]

        # Konwersja DF na numpy array
        features_objects = features_objects.to_numpy()
        features_numbers = features_numbers.to_numpy()
        target = target.to_numpy()

        # Kodowanie One Hot Encoding jeśli wymagane
        if features_objects.any():
            features_objects = data.enc_ohe_features.fit_transform(
                features_objects
            )
            data.decode = 1
        else:
            data.decode = 0

        features = np.concatenate((features_numbers, features_objects), axis=1)

        # Podzielenie tablic na zawierające NaN w wybranej kolumnie i
        # wypełnione
        data.features_all_nan = features[last_full_id + 1 :, :]
        data.features_no_nan = features[: last_full_id + 1, :]

        data.target_all_nan = target[last_full_id + 1 :]
        data.target_no_nan = target[: last_full_id + 1]

        # Usunięcie nazwy wypełnianej kolumny z listy
        del data.cols_to_fill[0]

    # Przywrócenie danym ich pierwotnej formy
    @staticmethod
    def revert_categorical(data, col):
        # Przywrócenie danym kategorycznym odpowiednich wartości
        data.features_no_nan = data.enc_label_features_no.inverse_transform(
            data.features_no_nan
        )
        data.target_no_nan = data.enc_label_target.inverse_transform(
            data.target_no_nan.reshape(-1, 1)
        ).ravel()
        data.features_all_nan = data.enc_label_features_all.inverse_transform(
            data.features_all_nan
        )
        data.target_all_nan = data.enc_label_target.inverse_transform(
            data.target_all_nan.reshape(-1, 1)
        ).ravel()

        # Zamiana Numpy Array na DataFrame
        data.features_all_nan = pd.DataFrame(data.features_all_nan)
        data.target_all_nan = pd.Series(data.target_all_nan, name="target")
        data.features_no_nan = pd.DataFrame(data.features_no_nan)
        data.target_no_nan = pd.Series(data.target_no_nan, name="target")

        data.df_all_nan = data.features_all_nan.join(data.target_all_nan)
        data.df_no_nan = data.features_no_nan.join(data.target_no_nan)

        data.df = pd.concat([data.df_all_nan, data.df_no_nan])

        data.df.columns = data.column_names

        # Przywrócenie kolumnom właściwych typów
        data.df = data.df.convert_dtypes(convert_string=False)

    @staticmethod
    def revert_numerical(data, col):
        # Przywrócenie danym kategorycznym odpowiednich wartości
        data.features_all_nan_objects = data.features_all_nan[
            :, len(data.features_numbers_names) :
        ]
        data.features_all_nan_numbers = data.features_all_nan[
            :, : len(data.features_numbers_names)
        ]
        data.features_no_nan_objects = data.features_no_nan[
            :, len(data.features_numbers_names) :
        ]
        data.features_no_nan_numbers = data.features_no_nan[
            :, : len(data.features_numbers_names)
        ]

        if data.decode:
            data.features_all_nan_objects = (
                data.enc_ohe_features.inverse_transform(
                    data.features_all_nan_objects
                )
            )
            data.features_no_nan_objects = (
                data.enc_ohe_features.inverse_transform(
                    data.features_no_nan_objects
                )
            )

        data.features_all_nan = np.concatenate(
            (data.features_all_nan_numbers, data.features_all_nan_objects),
            axis=1,
        )
        data.features_no_nan = np.concatenate(
            (data.features_no_nan_numbers, data.features_no_nan_objects),
            axis=1,
        )

        data.features_no_nan = pd.DataFrame(data.features_no_nan)
        data.target_no_nan = pd.Series(data.target_no_nan, name="target")

        data.features_all_nan = pd.DataFrame(data.features_all_nan)
        data.target_all_nan = pd.Series(data.target_all_nan, name="target")

        if col in data.cols_with_int:
            for i in data.target_all_nan:
                data.target_all_nan[i] = round(data.target_all_nan[i])

        # Łączenie tabel, przywracając oryginalną tabelę
        data.df_all_nan = data.features_all_nan.join(data.target_all_nan)
        data.df_no_nan = data.features_no_nan.join(data.target_no_nan)

        data.df = pd.concat([data.df_all_nan, data.df_no_nan])

        data.df.columns = data.column_names

        # Przywrócenie kolumnom właściwych typów
        data.df = data.df.convert_dtypes(convert_string=False)

        data.df = data.temp_filler.revert_nan(data.df)


fill_nan(Data())
