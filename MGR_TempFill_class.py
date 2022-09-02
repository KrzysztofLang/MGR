import numpy as np
import pandas as pd
from sympy import false, true


class TempFiller:
    def __init__(self) -> None:
        self.journal = pd.DataFrame(columns=['column','row'])


    def temp_fill(self, data):
        print("Rozwaza kolumne: " + data[0])
        for items in data[1].iteritems():
            if pd.isna(items[1]):
                self.journal.loc[len(self.journal)] = [data[0], items[0]]

        if data[1].dtype == "object" or data[1].dtype == "category":
            print("Wypelnia kategorie, kolumna: " + data[0])
            data[1].fillna(data[1].mode(), inplace = True)
        else:
            print("Wypelnia liczby, kolumna: " + data[0])
            data[1].fillna(data[1].mean(), inplace = True)

        return data[1]

    def revert_nan(self, data):
        pass

test = TempFiller()
print(test.journal)