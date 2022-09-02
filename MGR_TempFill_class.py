import numpy as np
import pandas as pd
from sympy import false, true


class TempFiller:
    def __init__(self) -> None:
        self.journal = pd.DataFrame(columns=['column','row'])


    def temp_fill(self, data):
        na_flag = false
        for items in data[1].iteritems():
            if pd.isna(items[1]):
                na_flag = true
                self.journal.loc[len(self.journal)] = [data[0], items[0]]
        if na_flag:
            if data[1].dtype == "object" or data[1].dtype == "category":
                data[1].fillna(data[1].mode())
            else:
                data[1].fillna(data[1].mean())

        return data

    def revert_nan(self, data):
        pass

test = TempFiller()
print(test.journal)