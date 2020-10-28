import os
from ShapeAssembly import Program
from utils import loadHPFromFile, writeHierProg

# files = os.listdir("random_data")
# P = Program()
#
# for f in files:
#     with open(f"random_data/{f}", "r") as file:
#         lines = file.readlines()
#     new_lines = []
#     for l in lines:
#         if "Cuboid" in l:
#             parse = P.parseCuboid(l[1:-1])
#             new_num = [round(x.item(), 3) for x in parse[1:4]]
#             new_lines.append(f" {parse[0]} = Cuboid({new_num[0]}, {new_num[1]}, {new_num[2]}, {parse[4]})\n")
#         elif "attach" in l:
#             parse = P.parseAttach(l)
#             new_num = [round(x.item(), 3) for x in parse[2:]]
#             new_lines.append(f" attach({parse[0]}, {parse[1]}, {new_num[0]}," +
#                                       f" {new_num[1]}, {new_num[2]}, {new_num[3]}, {new_num[4]}, {new_num[5]})\n")
#         else:
#             new_lines.append(l)
#     with open(f"random_data_fixed/{f}", "w") as file:
#         for l in new_lines:
#             file.write(l)

files = os.listdir("random_hier_data")
P = Program()

for f in files:
    prog = loadHPFromFile(f"random_hier_data/{f}")
    def fix_lines(prog):
        def order(l):
            if "Cuboid(" in l:
                name = P.parseCuboid(l)[0]
                if name == "bbox":
                    return 0
                else:
                    return int(name[4:]) + 1
            elif ("reflect" in l) or ("translate" in l):
                return 1000
            else:
                return 100
        prog['prog'].sort(key=order)
        for c in prog['children']:
            if not c == {}:
                fix_lines(c)
    fix_lines(prog)
    writeHierProg(prog, f"random_hier_data_fixed/{f}")
