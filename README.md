# Praca magisterska - Implementacja wybranych algorytmów wypełniania brakujących wartości, dla strumieni dużych zbiorów danych
## Autor

Krzysztof Lang

## Założenia pracy

Implementacja wybranych algorytmów wypełniania brakujących wartości, dla strumieni dużych zbiorów danych
W przypadku brakujących wartości w danych (potocznie: NULL), niektóre analizy danych (w szczególności: niektóre statystyki), mogą dawać zaburzone wyniki. Stąd, występuje podejście wypełniania (ang. imputation) brakujących wartości w danych, polegające na zastępowaniu brakujących wartości, pełnoprawnymi wartościami dla danego atrybutu. Jeśli algorytmy analizujące dane lub je przetwarzające, nie są dostosowane do radzenia sobie z brakującymi wartościami, należy ten problem zaadresować na etapie ETL (ściślej: czyszczenia danych). Problem ten jest tym bardziej aktualny, jeśli weźmiemy pod uwagę analizy dużych zbiorów danych (ang. BigData), gdyż w takim przypadku, dochodzą dodatkowo do głosu, aspekty złożonościowe algorytmów wypełniania. Praca magisterska będzie polegała na zaimplementowaniu kilku prostych algorytmów wypełniających brakujące wartości, dla dużych zbiorów danych, a także, na implementacji zaawansowanej heurystyki wielokrotnego wypełniania (ang. multiple imputation, (Rubin, 1996)) i/lub stworzenia klasyfikatora, którego zadaniem jest wypełnienie brakujących wartości, w rozumieniu działającego systemu decyzyjnego (opisanego szerzej w (Pietal, 2020)).

## O projekcie

Wg aktualnych założeń, program ma wypełniać puste wartości w zbiorze danych, niezależnie od typu danych i ilości kolumn w których występują. Stosowany algorytm opiera się na rozwiązywaniu problemów decyzyjnych.

Wykorzystywany jest Python 3.10.5.

Główny plik realizujący założenia projektu to *MGR_main.py*, wraz z modułem *MGR_learn_fill*. Dodatkowo *NaN_gen.py* służy do przygotowywania plików z danymi usuwając losowo wartości.

