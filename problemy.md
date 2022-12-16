# Problemy z kodem

## Za dużo kolumn po kodowaniu numerycznych

MGR_Data_class.py, 172:

    features = self.enc_ohe_features.fit_transform(features)
    print(np.shape(features)[1])

    50821

W przeciwieństwie do 'OrdinalEncoder', 'OneHotEncoder' nie koduje wyłączenie danych katgorycznych, tylko wszystkie w podanym zbiorze danych.  
Potrzebny jest sposób na takie wydzielenie danych kategorycznych, zakodowanie ich, a następnie ponowne połączenie z resztą danych, tak aby móc cofnąć kodowanie po uczeniu i nie wywołać błędów w innych miejscach kodu.  
To oznacza:

- Aby móc cofnąć kodowanie, dane muszą wyglądać identycznie jak po zakodowaniu
- Aby przywrócić format DataFrame z Numpy, konieczna jest znajomość kolejności kolumn