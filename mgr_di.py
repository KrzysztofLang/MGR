class DownImpu:
    def __init__(self) -> None:
        self.cols_count_nan = dict()
        self.cols_list = dict()

    def Prepare(self, data):
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
            for col in self.cols_count_nan:
                # Kolumna z największą ilością wspólnych NaN z poprzednią
                most_common = list(self.cols_count_nan.keys())[0]
                # Ile jest najwięcej wspólnych NaN między sprawdzanymi kolumnami
                most_com_nans = 0
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

        print("Lista kolumn:\n", self.cols_list)
        exit()
