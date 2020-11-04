import os

files = os.listdir("kenny_data")

for f in files:
    if f[:5] == "prog_":
        with open(f"kenny_data/{f}", "r") as file:
            lines = file.readlines()
        lines = [f"\t{l}" for l in lines]
        lines = ["Assembly Program_0 {\n"] + lines + ["\t}"]
        with open(f"kenny_data/{f[5:]}", "w") as file:
            for l in lines:
                file.write(l)
