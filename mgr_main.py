from easygui import *

import mgr_fill
import mgr_nan_gen
import mgr_acc

choices = ["NaN Filler", "NaN Generator", "Accuracy Test"]
match choicebox("Wybierz program:", "NaN Suite", choices):
    case "NaN Filler":
        dum = mgr_fill.Fill()
    case "NaN Generator":
        dum = mgr_nan_gen.NanGen()
    case "Accuracy Test":
        dum = mgr_acc.AccuracyTester()
