from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from easygui import *

import mgr_data
import mgr_di


class Fill:
    def __init__(self) -> None:
        self.fill_nan(mgr_data.Data())

    def check_datatype(self, data):
        # Wybranie kolumny do wypełnienia
        col = data.cols_to_fill[0]

        # Sprawdzenie typu danych
        if (
            data.df[col].dtypes == "object"
            or data.df[col].dtypes == "category"
        ):
            type = "cat"
        else:
            type = "num"

        return type, col

    def fill_categorical(self, data, col):
        # Wydzielenie zbiorów uczących i testowych
        x_train, x_test, y_train, y_test = train_test_split(
            data.features_no_nan,
            data.target_no_nan,
            test_size=0.3,
            random_state=109,
        )

        # Nauka modelu
        clf = HistGradientBoostingClassifier(
            max_iter=100, categorical_features=data.cat_arr
        ).fit(x_train, y_train)

        # Test skuteczności modelu
        y_pred = clf.predict(x_test)

        data.target_all_nan = clf.predict(data.features_all_nan)

    def fill_numerical(self, data, col):
        # Wydzielenie zbiorów uczących i testowych
        x_train, x_test, y_train, y_test = train_test_split(
            data.features_no_nan[:, 1:],
            data.target_no_nan,
            test_size=0.3,
            random_state=0,
        )

        # Nauka modelu
        model = LinearRegression()
        model.fit(x_train, y_train)

        data.target_all_nan = model.predict(data.features_all_nan[:, 1:])

    @staticmethod
    def save(data):
        data.df = data.df[data.columns]
        name = (
            data.file[2 : len(data.file) - 4]
            + "_filled_"
            + data.algorithm
            + ".csv"
        )
        data.df.to_csv(
            name,
            index=False,
        )
        msgbox(
            "Zakończono wypełnianie.\nWynik zapisano do pliku " + name,
            "Wypełnianie pustych miejsc",
        )
        exit()

    def progress_up(self):
        self.progress += self.chunk
        print(
            "Postęp wypełniania: ",
            round(self.progress),
            "%",
        )

    # Główna funkcja
    def fill_nan(self, data):
        self.chunk = len(data.cols_to_fill) * 3
        self.chunk = 100 / self.chunk
        self.progress = 0
        print("Postęp wypełniania: ", round(self.progress), "%")
        match data.algorithm:
            case "Prosty":
                while True:
                    if data.cols_to_fill:
                        type, col = self.check_datatype(data)
                        print("Wypełnianie kolumny", col)
                        match type:
                            case "num":
                                mgr_data.PrepareData.prepare_numerical(
                                    data, col
                                )

                                self.progress_up()

                                self.fill_numerical(data, col)

                                self.progress_up()

                                mgr_data.PrepareData.revert_numerical(
                                    data, col
                                )

                                self.progress_up()

                            case "cat":
                                mgr_data.PrepareData.prepare_categorical(
                                    data, col
                                )

                                self.progress_up()

                                self.fill_categorical(data, col)

                                self.progress_up()

                                mgr_data.PrepareData.revert_categorical(
                                    data, col
                                )

                                self.progress_up()
                    else:
                        self.save(data)
            case "Downward Imputation":
                di = mgr_di.DownImpu()
                di.prime(data)
                while True:
                    if data.cols_to_fill:
                        type, col = self.check_datatype(data)
                        print("Wypełnianie kolumny", col)
                        match type:
                            case "num":
                                di.prepare(col, data)
                                mgr_data.PrepareData.prepare_numerical(
                                    data, col
                                )

                                self.progress_up()

                                self.fill_numerical(data, col)

                                self.progress_up()

                                mgr_data.PrepareData.revert_numerical(
                                    data, col
                                )

                                self.progress_up()

                                data.df = di.temp_df.join(data.df)
                            case "cat":
                                di.prepare(col, data)
                                mgr_data.PrepareData.prepare_categorical(
                                    data, col
                                )

                                self.progress_up()

                                self.fill_categorical(data, col)

                                self.progress_up()

                                mgr_data.PrepareData.revert_categorical(
                                    data, col
                                )

                                self.progress_up()

                                data.df = di.temp_df.join(data.df)

                    else:
                        self.save(data)
            case "Uproszczony Downward Imputation":
                di = mgr_di.DownImpu()
                while True:
                    if data.cols_to_fill:
                        type, col = self.check_datatype(data)
                        print("Wypełnianie kolumny", col)
                        match type:
                            case "num":
                                di.prepare(col, data)
                                mgr_data.PrepareData.prepare_numerical(
                                    data, col
                                )

                                self.progress_up()

                                self.fill_numerical(data, col)

                                self.progress_up()

                                mgr_data.PrepareData.revert_numerical(
                                    data, col
                                )

                                self.progress_up()

                                data.df = di.temp_df.join(data.df)
                            case "cat":
                                di.prepare(col, data)
                                mgr_data.PrepareData.prepare_categorical(
                                    data, col
                                )

                                self.progress_up()

                                self.fill_categorical(data, col)
                                mgr_data.PrepareData.revert_categorical(
                                    data, col
                                )

                                self.progress_up()

                                data.df = di.temp_df.join(data.df)

                                self.progress_up()
                    else:
                        self.save(data)
