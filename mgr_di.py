class DownImpu:
    def __init__(self) -> None:
        self.cols_count_nan = dict()

    def Prepare(self, data):
        # Tworzenie słownika
        # gdzie klucz to nazwa kolumny z NaN a wartość to lista ID z NaN
        for col in data.cols_to_fill:
            temp = data.df[col]
            self.cols_count_nan[col] = temp[temp.isna()].index.tolist()

        exit()
