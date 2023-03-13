import pandas as pd


class TempFill:
    def __init__(self) -> None:
        self.journal = pd.DataFrame(columns=["row", "column"])

    def temp_fill(self, data):
        for items in data[1].items():
            if pd.isna(items[1]):
                self.journal.loc[len(self.journal)] = [items[0], data[0]]

        if data[1].dtype == "object" or data[1].dtype == "category":
            filler = data[1].mode()
            filler = filler[0]
            data[1].fillna(filler, inplace=True)
        else:
            filler = data[1].mean()
            data[1].fillna(filler, inplace=True)

        return data[1]

    def revert_nan(self, data):
        for items in self.journal.itertuples(index=False):
            data.loc[items[0], items[1]] = None

        return data
