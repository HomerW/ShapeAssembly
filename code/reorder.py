from ShapeAssembly import make_hier_prog
from utils import writeHierProg
import os
from copy import copy

def print_recur(prog):
    lines = 0
    for p in prog['prog']:
        print(p)
        lines += 1
    for c in prog['children']:
        if not c == {}:
            lines += print_recur(c)
    return lines

# progs = os.listdir("parse_flat_chair/")
#
# for p in progs:
#     with open(f"parse_flat_chair/{p}", "r") as file:
#         lines = file.readlines()
#     p_dict = make_hier_prog(lines)
#     cuboid_lines = [x for x in p_dict['prog'] if "Cuboid" in x]
#     non_cuboid_lines = [x for x in p_dict['prog'] if not "Cuboid" in x]
#     new_lines = copy(non_cuboid_lines)
#     for line in cuboid_lines:
#         name = line.split("=")[0][:-1]
#         relevant_lines = [x for x in non_cuboid_lines if name in x]
#         insert_idx = new_lines.index(relevant_lines[0])
#         new_lines.insert(insert_idx, line)
#     p_dict['prog'] = new_lines
#     writeHierProg(p_dict, f"reorder/{p}")

progs = os.listdir("data/chair/")

for p in progs:
    with open(f"data/chair/{p}", "r") as file:
        lines = file.readlines()
    p_dict = make_hier_prog(lines)
    def reorder_recur(prog):
        cuboid_lines = [x for x in prog['prog'] if "Cuboid" in x]
        non_cuboid_lines = [x for x in prog['prog'] if not "Cuboid" in x]
        new_lines = copy(non_cuboid_lines)
        for line in cuboid_lines:
            name = line.split("=")[0][:-1]
            relevant_lines = [x for x in non_cuboid_lines if name in x]
            insert_idx = new_lines.index(relevant_lines[0])
            new_lines.insert(insert_idx, line)
        prog['prog'] = new_lines
        new_cuboid_lines = [x for x in new_lines if "Cuboid" in x]
        new_order = [new_cuboid_lines.index(x) for x in cuboid_lines]
        temp_children = sorted(zip(new_order, prog['children']))
        prog['children'] = [x[1] for x in temp_children]
        for child in prog['children']:
            if not child == {}:
                reorder_recur(child)
    reorder_recur(p_dict)
    writeHierProg(p_dict, f"data/reorder/{p}")
