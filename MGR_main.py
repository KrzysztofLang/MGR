import MGR_class as cl
import MGR_learn_fill as lf

def napis():

    print(" _   _       _   _   ______ _ _ _ ")
    print("| \\ | |     | \\ | | |  ____(_) | |")
    print("|  \\| | __ _|  \\| | | |__   _| | | ___ _ __ ")
    print("| . ` |/ _` | . ` | |  __| | | | |/ _ \\ '__|")
    print("| |\\  | (_| | |\\  | | |    | | | |  __/ |")
    print("|_| \\_|\\__,_|_| \\_| |_|    |_|_|_|\\___|_|")
    print('Aby wyjść z programu, wpisz "koniec".\n')

napis()

dane = cl.Dane()

lf.naucz_i_wypełnij(dane)
