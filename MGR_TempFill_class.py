import numpy as np
import pandas as pd
from sympy import false, true


class TempFiller:
    def __init__(self) -> None:
        pass

    @classmethod
    def temp_fill(cls, data):
        if data.dtype == "object" or data.dtype == "category":
            cls.temp_fill_categorical(data)
        else:
            cls.temp_fill_numerical(data)

    def temp_fill_categorical(self, data):
        pass

    def temp_fill_numerical(self, data):
        pass

    def revert_nan(self, data):
        pass