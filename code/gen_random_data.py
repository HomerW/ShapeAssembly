import networkx as nx
import random
from ShapeAssembly import Program
from itertools import product
import torch
import intersect_modified as inter
from copy import deepcopy, copy
import matplotlib.pyplot as plt
import numpy as np
from valid import check_stability

def rand_program2():
    size_min = 0.05
    size_max = 0.5
    align_prob = 0.70
    max_sources = 4
    new_cuboid_prob = 0.5
    max_out = 3
    max_att_overlap = 0.15
    max_disc_overlap = 0.00
    max_cuboids = max(2, min(10, int(random.normalvariate(7, 2))))

    def valid(P):
        cuboid_objects = [v for k, v in P.cuboids.items() if k != 'bbox']
        good_overlap = False
        if len(cuboid_objects) == 1:
            good_overlap = True
        else:
            overlap, attachments = inter.findOverlapAmount(cuboid_objects)
            att_overlap = [o for o, a in zip(overlap, attachments) if a]
            disc_overlap = [o for o, a in zip(overlap, attachments) if not a]
            if att_overlap == [] and not disc_overlap == []:
                good_overlap = max(disc_overlap) <= max_disc_overlap
            elif disc_overlap == [] and not att_overlap == []:
                good_overlap = max(att_overlap) <= max_att_overlap
            else:
                good_overlap = max(att_overlap) <= max_att_overlap and max(disc_overlap) <= max_disc_overlap
        corners = []
        for c in cuboid_objects:
            corners.append(c.getCorners())
        corners = torch.stack(corners)
        good_corners = (corners >= -0.5).all() and (corners <= 0.5).all()
        # if not good_overlap:
        #     print("TOO MUCH OVERLAP")
        # if not good_corners:
        #     print("OUTSIDE BOX")
        return good_overlap and good_corners

    def attach_source(src_cuboid, P):
        counter = 0
        while counter < 100:
            old_prog = deepcopy(P)
            # bottom of bbox and bottom of (unattached) source cuboid
            attach2 = [random.random(), 0, random.random()]
            attach1 = [random.random(), 0, random.random()]
            size = [random.uniform(size_min, size_max) for _ in range(3)]
            alignment = (random.random() < align_prob)
            lines = [
                f"cube{src_cuboid['id']} = Cuboid({size[0]}, {size[1]}, {size[2]}, {alignment})",
                f"attach(cube{src_cuboid['id']}, bbox, " +
                f"{attach1[0]}, {attach1[1]}, {attach1[2]}, " +
                f"{attach2[0]}, {attach2[1]}, {attach2[2]})"
            ]
            for l in lines:
                P.execute(l)
            if valid(P):
                break
            P = old_prog
            counter += 1
        if counter >= 100:
            print("GAVE UP ON SOURCE")
        return P, lines

    def create_edge(src_cuboid, cuboids, P, new_src_cuboids, prev_attach, new_cuboid):
        lines = []
        faces = ['right', 'left', 'top', 'bot', 'front', 'back']
        face = faces[random.choice(range(6))]
        _, ind, val = P.getFacePos(face)
        # pick random point on random face of first cuboid
        attach1 = [random.random() for _ in range(2)]
        attach1 = attach1[:ind] + [val] + attach1[ind:]
        face = faces[random.choice(range(6))]
        _, ind, val = P.getFacePos(face)
        # pick random point on random face of second cuboid
        attach2 = [random.random() for _ in range(2)]
        attach2 = attach2[:ind] + [val] + attach2[ind:]

        new_attach = None
        possible_attachments = []
        for c in cuboids:
            if not (c["id"] in src_cuboid["ancestors"] or c["level"] == 0 or c['id'] in prev_attach):
                possible_attachments.append(c)
        if new_cuboid or possible_attachments == []:
            if len(P.cuboids) >= max_cuboids:
                print("TOO MANY CUBOIDS")
                return cuboids, P, new_src_cuboids, [], new_attach
            size = [random.uniform(size_min, size_max) for _ in range(3)]
            alignment = (random.random() < align_prob)
            if new_src_cuboids == []:
                index = max([x['id'] for x in cuboids]) + 1
            else:
                index = max([x['id'] for x in new_src_cuboids]) + 1
            new_src_cuboids.append({"id": index,
                                    "ancestors": src_cuboid["ancestors"] | set([index]),
                                    "level": src_cuboid["level"] + 1})
            lines = [f"cube{index} = Cuboid({size[0]}, {size[1]}, {size[2]}, {alignment})"]
            P.execute(lines[0])
            c_new = f"cube{index}"
        else:
            # pos_levels = [c['level'] for c in possible_attachments]
            # p = [-1*(x - max(pos_levels))+1.0 for x in pos_levels]
            # p = p / np.sum(p)
            selected_cuboid = np.random.choice(possible_attachments)
            selected_cuboid['ancestors'] = selected_cuboid['ancestors'] | src_cuboid['ancestors']
            c_new = f"cube{selected_cuboid['id']}"
            new_attach = selected_cuboid['id']

        lines.append(f"attach({c_new}, cube{src_cuboid['id']}, "
                  f"{attach1[0]}, {attach1[1]}, {attach1[2]}, "
                  f"{attach2[0]}, {attach2[1]}, {attach2[2]})")
        P.execute(lines[-1])

        return cuboids, P, new_src_cuboids, lines, new_attach

    def add_attachment(src_cuboid, cuboids, P, num_sources):
        merged = True
        if num_sources > 1:
            num_source_anc = len([x for x in src_cuboid['ancestors'] if x < num_sources])
            if num_source_anc <= 1:
                merged = False
        if merged:
            #reduce prob of 0 and 3 outgoing attachments
            p = np.full((max_out + 1,), 1.0)
            p[0] = p[0] / 5
            p[3] = p[3] / 5
            p[2] = p[2] / 2
            p = p / np.sum(p)
            out = np.random.choice(list(range(max_out+1)), p=p)
        else:
            #reduce prob of 3 outgoing attachments
            p = np.full((max_out,), 1.0)
            p[2] = p[2] / 5
            p[1] = p[1] / 2
            p = p / np.sum(p)
            out = np.random.choice(list(range(1, max_out+1)), p=p)
            # out = np.random.choice(list(range(1, max_out+1)))
        print(f"OUT: {out}")

        new_src_cuboids = []
        lines = []
        prev_attach = []

        for edge in range(out):
            edge_counter = 0
            new_cuboid = random.random() < new_cuboid_prob
            while edge_counter < 100:
                old_prog = deepcopy(P)
                old_new_src_cuboids = deepcopy(new_src_cuboids)
                old_cuboids = deepcopy(cuboids)
                old_lines = deepcopy(lines)
                old_prev_attach = deepcopy(prev_attach)

                cuboids, P, new_src_cuboids, new_lines, new_attach = create_edge(src_cuboid, cuboids, \
                                                                                 P, new_src_cuboids, prev_attach, new_cuboid)
                prev_attach.append(new_attach)
                lines += new_lines

                if valid(P):
                    if not new_cuboid:
                        print("MADE ATTACHMENT WITH PREV CUBOID")
                        global prev_att
                        prev_att += 1
                    else:
                        print("MADE NEW CUBOID")
                        global new_cub
                        new_cub += 1
                    global edge_count
                    edge_count += 1
                    break
                P = old_prog
                new_src_cuboids = old_new_src_cuboids
                cuboids = old_cuboids
                lines = old_lines
                prev_attach = old_prev_attach

                edge_counter += 1

            if not out == 0 and edge_counter >= 100:
                print("GAVE UP ON EDGE")

        # remove cuboids close to bbox ceiling
        filtered_src_cuboids = []
        for c in new_src_cuboids:
            corner_heights = P.cuboids[f"cube{c['id']}"].getCorners()[:, 1]
            if not (corner_heights >= 0.4).any():
                filtered_src_cuboids.append(c)
            else:
                print("MADE A SINK CUBOID")

        return new_src_cuboids, filtered_src_cuboids, P, lines

    P = Program()
    lines = []
    num_source = random.choice(list(range(1, max_sources+1)))
    current_cuboids = []
    for source in range(num_source):
        current_cuboids.append({"id": source,
                                "ancestors": set([source]),
                                "level": 0})
    for source in current_cuboids:
        P, new_lines = attach_source(source, P)
        lines += new_lines
    cuboids = current_cuboids
    while not (current_cuboids == [] or len(P.cuboids) >= max_cuboids):
        new_current_cuboids = []
        for c in current_cuboids:
            new_src_cuboids, filtered_src_cuboids, P, new_lines = add_attachment(c, cuboids, P, num_source)
            lines += new_lines
            new_current_cuboids += filtered_src_cuboids
            cuboids += new_src_cuboids
        current_cuboids = new_current_cuboids

    if len(P.cuboids) > max_cuboids:
        print("Terminated because hit cuboid limit")
    else:
        print("Terminated because no more cuboids to attach")

    lines = ["bbox = Cuboid(1.0, 1.0, 1.0, True)"] + lines

    return P, lines

edge_count = 0
new_cub = 0
prev_att = 0

def canonical(lines):
    P = Program()
    def order(l):
        if "Cuboid(" in l:
            name = P.parseCuboid(l)[0]
            if name == "bbox":
                return 0
            else:
                return int(name[4:]) + 1
        elif ("squeeze" in l) or ("reflect" in l) or ("translate" in l):
            return 1000
        else:
            return 100
    lines.sort(key = order)
    lines = [" " + l for l in lines]
    lines = ["Assembly Program_0 {"] + lines + ["}"]
    return lines

for n in range(1000, 1001):
    prog, lines = rand_program2()
    while not check_stability(*prog.getShapeGeo()):
        prog, lines = rand_program2()
    # prog.render(ofile=f"{n}.obj")
    lines = canonical(lines)
    with open(f"random_data/{n}.txt", "w") as file:
        for l in lines:
            file.write(f"{l}\n")
