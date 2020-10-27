import os
from ShapeAssembly import Program

files = os.listdir("random_data")
P = Program()

for f in files:
    with open(f"random_data/{f}", "r") as file:
        lines = file.readlines()
    new_lines = []
    for l in lines:
        if "Cuboid" in l:
            parse = P.parseCuboid(l[1:-1])
            new_num = [round(x.item(), 3) for x in parse[1:4]]
            new_lines.append(f" {parse[0]} = Cuboid({new_num[0]}, {new_num[1]}, {new_num[2]}, {parse[4]})\n")
        elif "attach" in l:
            parse = P.parseAttach(l)
            new_num = [round(x.item(), 3) for x in parse[2:]]
            new_lines.append(f" attach({parse[0]}, {parse[1]}, {new_num[0]}," +
                                      f" {new_num[1]}, {new_num[2]}, {new_num[3]}, {new_num[4]}, {new_num[5]})\n")
        else:
            new_lines.append(l)
    with open(f"random_data_fixed/{f}", "w") as file:
        for l in new_lines:
            file.write(l)
