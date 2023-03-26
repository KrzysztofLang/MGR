from easygui import *

import mgr_fill
import mgr_nan_gen
import mgr_acc

choices = [
    "Przygotowanie danych",
    "Wypełnianie pustych miejsc",
    "Test skuteczności wypełnienia",
]
match choicebox(
    "    _   __      _   __   _____       _ __\n"
    + "   / | / /___ _/ | / /  / ___/__  __(_) /____\n"
    + "  /  |/ / __ `/  |/ /   \__ \/ / / / / __/ _ \\\n"
    + " / /|  / /_/ / /|  /   ___/ / /_/ / / /_/  __/\n"
    + "/_/ |_/\__,_/_/ |_/   /____/\__,_/_/\__/\___/\n\n"
    + "Wybierz moduł:",
    "Krzysztof Lang",
    choices,
):
    case "Wypełnianie pustych miejsc":
        dum = mgr_fill.Fill()
    case "Przygotowanie danych":
        dum = mgr_nan_gen.NanGen()
    case "Test skuteczności wypełnienia":
        dum = mgr_acc.AccuracyTester()
