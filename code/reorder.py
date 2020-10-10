from ShapeAssembly import make_hier_prog
from utils import writeHierProg
import os
from copy import copy

progs = os.listdir("parse_flat_chair/")

for p in progs:
    with open(f"parse_flat_chair/{p}", "r") as file:
        lines = file.readlines()
    p_dict = make_hier_prog(lines)
    cuboid_lines = [x for x in p_dict['prog'] if "Cuboid" in x]
    non_cuboid_lines = [x for x in p_dict['prog'] if not "Cuboid" in x]
    new_lines = copy(non_cuboid_lines)
    for line in cuboid_lines:
        name = line.split("=")[0][:-1]
        relevant_lines = [x for x in non_cuboid_lines if name in x]
        insert_idx = new_lines.index(relevant_lines[0])
        new_lines.insert(insert_idx, line)
    p_dict['prog'] = new_lines
    writeHierProg(p_dict, f"reorder/{p}")
