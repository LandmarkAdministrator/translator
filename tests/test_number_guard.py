"""Tests for the number guard. No GPU, no models: the translator is faked."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from pipeline.number_guard import (numbers, keeps_numbers, segment_translate,
                                   guard, audit, tidy, present, missing,
                                   mask_translate)

fails, checks = [], []
def check(name, got, want):
    checks.append(name)
    if got != want:
        fails.append(f"{name}: got {got!r}, wanted {want!r}")

check("finds plain numbers", numbers("Daniel chapter 6 verse 10"), ["6", "10"])
check("keeps time and refs intact", numbers("a scan at 7:30 and John 3:16"), ["7:30", "3:16"])
check("strips thousands separators", numbers("2,500 dollars"), ["2500"])
check("no numbers", numbers("Thank you, brother."), [])
check("match", keeps_numbers("hymn 413", "kantik 413"), True)
check("mangled number is not a match", keeps_numbers("hymn 413", "kantik 4 1513"), False)
check("dropped number is not a match", keeps_numbers("at 7:30", "nan maten"), False)
check("reordering is allowed, the numbers are what matter", keeps_numbers("6 then 10", "10 then 6"), True)
check("a number going missing is not", keeps_numbers("6 then 10", "10 alone"), False)

# A number may legitimately change shape; only disappearing counts as a loss.
check("year shortened to two digits", present("1970", "a mediados de los años 70", "es"), True)
check("Russian year shortened", present("1970", "в середине 70-х", "ru"), True)
check("time split into parts", present("7:30", "nan 7 è 30 nan maten", "ht"), True)
check("spelled out in Spanish", present("413", "himno cuatrocientos trece", "es"), True)
check("spelled out in Creole", present("6", "Danyèl chapit sis", "ht"), True)
check("mangled number is still a loss", present("413", "kantik 4 1513", "ht"), False)
check("vanished number is a loss", present("30", "nan tren", "ht"), False)
check("missing lists only the losses", missing("hymn 413 at 7:30", "kantik 4 1513 nan 7 è 30", "ht"), ["413"])

# Placeholders: used for Spanish and Russian, where they survive.
import re as _re
def masks_fine(text):          # keeps X0-style markers, as Spanish models do
    return "[" + text + "]"
def drops_bare_numbers(text):  # drops loose digits but respects placeholders
    return _re.sub(r"(?<![A-Za-z])\d+(?:[.,:/]\d+)*", "", text).strip()
out = mask_translate("Turn to hymn 413 and 6.", masks_fine, "es")
check("masking restores every number", numbers(out), ["413", "6"])

# A translator that drops every number, as Opus does in Creole.
def drops_numbers(text):
    import re
    return "[" + re.sub(r"\d+(?:[.,:/]\d+)*", "", text).strip() + "]"

out, how = guard("Turn to hymn 413.", drops_numbers, "ht")
check("repairs a dropped number", numbers(out), ["413"])
check("reports the repair", how, "repaired")
out, how = guard("Turn to hymn 413.", drops_bare_numbers, "es")
check("Spanish is masked rather than cut apart", how, "masked")
check("and the number is back", numbers(out), ["413"])

def faithful(text):
    return "[" + text + "]"
out, how = guard("Turn to hymn 413.", faithful, "ht")
check("leaves a good translation alone", how, "clean")
out, how = guard("Thank you, brother.", faithful, "ht")
check("no numbers to guard", how, "none")

def idiomatic(text):        # renders 1970 the way a translator would
    return text.replace("1970s", "los años 70")
out, how = guard("It was the mid-1970s.", idiomatic, "es")
check("an idiomatic year is left alone", how, "clean")
check("and not rewritten", "los años 70" in out, True)

out = segment_translate("Take your Bibles and go to Daniel chapter 6.", drops_numbers)
check("number survives segmentation", numbers(out), ["6"])
out = segment_translate("413. Fill my cup.", drops_numbers)
check("number at the start survives", numbers(out), ["413"])
out = segment_translate("Romans 8, verses 28 through 30.", drops_numbers)
check("several numbers survive", numbers(out), ["8", "28", "30"])
out = segment_translate("I had a scan at 7:30 in the morning.", lambda t: t.rstrip(".") + ".")
check("a fragment does not end the sentence early", out.count("."), 1)

check("tidies spacing", tidy("Chapit  6 ."), "Chapit 6.")
check("capitalizes", tidy("chapit 6."), "Chapit 6.")

a = audit([("hymn 413", "kantik 4 1513"), ("chapter 6", "chapit 6"), ("no numbers", "anyen")], "ht")
check("audit counts sentences with numbers", a["sentences_with_numbers"], 2)
check("audit counts losses", a["lost"], 1)
check("audit retention", round(a["retention"], 2), 0.5)

if fails:
    print("FAILED:"); [print("  " + f) for f in fails]; raise SystemExit(1)
print(f"OK — {len(checks)} checks passed")
