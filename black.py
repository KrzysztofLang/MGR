import subprocess
import glob

files = glob.glob("./*.py")

for file in files:
    print("\nPoprawiam plik ", file, ": ")
    exec = subprocess.Popen(
        [
            "powershell",
            "& {black --line-length 79 " + file + "}",
        ]
    )
    exec.wait()
