class DownImpu:
    def __init__(self) -> None:
        self.cols_count_nan = dict()
        self.cols_list = dict()

    def prime(self, data):
        # Tworzenie słownika
        # gdzie klucz to nazwa kolumny z NaN a wartość to lista ID z NaN
        for col in data.cols_to_fill:
            temp = data.df[col]
            self.cols_count_nan[col] = temp[temp.isna()].index.tolist()

        # Ilość NaN w kolumnie z największą ich ilością
        nans = 0
        # Kolumna z największą ilością NaN
        most_nans = 0

        # Znalezienie kolumny z największą ilością NaN
        for col in self.cols_count_nan:
            if len(self.cols_count_nan[col]) > nans:
                nans = len(self.cols_count_nan[col])
                most_nans = col
        self.cols_list[most_nans] = nans
        last = self.cols_count_nan.pop(most_nans)

        for i in range(len(self.cols_count_nan)):
            # Ile jest najwięcej wspólnych NaN między sprawdzanymi kolumnami
            most_com_nans = 0
            # Kolumna z największą ilością wspólnych NaN z poprzednią
            most_common = list(self.cols_count_nan.keys())[0]

            for col in self.cols_count_nan:
                # Ile jest wspólnych NaN między aktualnie sprawdzanymi kol
                com_nans = len(
                    set(last).intersection(self.cols_count_nan[col])
                )

                if com_nans > most_com_nans:
                    most_com_nans = com_nans
                    most_common = col
                elif com_nans == most_com_nans:
                    if len(self.cols_count_nan[col]) > len(
                        self.cols_count_nan[most_common]
                    ):
                        most_com_nans = com_nans
                        most_common = col

            if most_com_nans == 0:
                for col in self.cols_count_nan:
                    if len(self.cols_count_nan[col]) > nans:
                        most_common = col

            self.cols_list[most_common] = most_com_nans
            last = self.cols_count_nan.pop(most_common)

        print(self.cols_list)

        # Standardowe wypełnienie kolumn z najmniejszą ilością NaN
        while True:
            col = list(self.cols_list.keys())[-1]
            if self.cols_list[col] == 0:
                if (
                    data.df[col].dtype == "object"
                    or data.df[col].dtype == "category"
                ):
                    filler = data.df[col].mode()
                    filler = filler[0]
                    data.df[col].fillna(filler, inplace=True)
                else:
                    filler = data.df[col].mean()
                    data.df[col].fillna(filler, inplace=True)
                self.cols_list.popitem()
            else:
                break

        temp = list(self.cols_list.keys())
        temp.reverse()
        data.cols_to_fill = temp

    def prepare(self, col, data):
        # Utworzenie DF z kolumnami nie biorącymi udziału w nauce
        self.temp_df = data.df[data.cols_to_fill]
        self.temp_df = self.temp_df.drop(col, axis=1)
        data.df = data.df.drop(self.temp_df.columns, axis=1)
