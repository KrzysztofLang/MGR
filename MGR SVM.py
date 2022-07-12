from typing import Type
import numpy as np
import pandas as pd
from sympy import true
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import RandomFourierFeatures

##Przygotowanie modelu
model = keras.Sequential(
    [
        keras.Input(shape=(14,2)),
        RandomFourierFeatures(
            output_dim=4096, scale=10.0, kernel_initializer="gaussian"
        ),
        layers.Dense(units=10),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.hinge,
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)

##Wczytanie pliku danych
df = pd.read_csv("adult.data")

##Zamiana typow danych na kategorie, a następnie zakodowanie jako dane numeryczne w nowym DF
colsObjects = df.columns[df.dtypes == "object"].tolist()
for col in colsObjects:
    df[col] = df[col].astype('category')
    dfCoded = df
    dfCoded[col] = dfCoded[col].cat.codes
    dfCoded.loc[dfCoded[col] == -1, col] = np.nan

##Zamiana na float dla ujednolicenia typu
dfCoded = dfCoded.astype(np.float64)

##Wybranie kolumny do wypełnienia
cols = dfCoded.columns.to_list()
print("Dostępne kolumny:")
print(cols)
while true:
    col = input("Etykieta kolumny do wypełnienia (aby anulować, wpisz \"koniec\"): ")
    if col in cols:
        print('Wpisano poprawnie')
        break
    elif col == 'koniec':
        exit()
    else:
        print('Nie ma takiej kolumny!')

##Podzielenie Dataframe na zawierające NaN w wybranej kolumnie i wypełnione
dfAllNan = dfCoded[dfCoded[col].isnull()]
dfNoNan = dfCoded[~dfCoded[col].isnull()]

dfNoNanTarget = dfNoNan.pop(col)
dfAllNanTarget = dfAllNan.pop(col)

model.fit(dfNoNan, dfNoNanTarget, epochs=15, batch_size=2, validation_split=0.2)