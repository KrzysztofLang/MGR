from MGR_class import Dane
from MGR_learn_fill import fill_nan


def header():

    print(" _   _       _   _   ______ _ _ _ ")
    print("| \\ | |     | \\ | | |  ____(_) | |")
    print("|  \\| | __ _|  \\| | | |__   _| | | ___ _ __ ")
    print("| . ` |/ _` | . ` | |  __| | | | |/ _ \\ '__|")
    print("| |\\  | (_| | |\\  | | |    | | | |  __/ |")
    print("|_| \\_|\\__,_|_| \\_| |_|    |_|_|_|\\___|_|")
    print('Aby wyjść z programu, wpisz "koniec".\n')

header()

dane = Dane()

fill_nan(dane)
