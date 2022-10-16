from subprocess import Popen

subprocesses = []

for i in range(2):
    subprocesses.append(Popen(
        ["venv/Scripts/python.exe", f"main.py", f"{i}"]
    ))
for el in subprocesses:
    el.wait()
