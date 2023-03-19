from easygui import *

import mgr_fill
import mgr_nan_gen
import mgr_acc

choices = ["NaN Filler", "NaN Generator", "Accuracy Test"]
match choicebox(
    "    _   __      _   __   _____       _ __\n"
    + "   / | / /___ _/ | / /  / ___/__  __(_) /____\n"
    + "  /  |/ / __ `/  |/ /   \__ \/ / / / / __/ _ \\\n"
    + " / /|  / /_/ / /|  /   ___/ / /_/ / / /_/  __/\n"
    + "/_/ |_/\__,_/_/ |_/   /____/\__,_/_/\__/\___/\n\n"
    + "Wybierz moduł:",
    "NaN Suite",
    choices,
):
    case "NaN Filler":
        dum = mgr_fill.Fill()
    case "NaN Generator":
        dum = mgr_nan_gen.NanGen()
    case "Accuracy Test":
        dum = mgr_acc.AccuracyTester()
