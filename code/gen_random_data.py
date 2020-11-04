import networkx as nx
import random
from ShapeAssembly import Program, hier_execute
from itertools import product
import torch
import intersect_modified as inter
from copy import deepcopy, copy
import matplotlib.pyplot as plt
import numpy as np
from valid import check_stability
from utils import writeHierProg, writeObj

align_prob = 0.7
max_sources = 4
new_cuboid_prob = 0.5
max_out = 3
max_att_overlap = 0.15
max_disc_overlap = 0.00

def samp(mean, std, _min, _max):
    return torch.clamp((torch.randn(1) * std) + mean, _min, _max)

def samp_dims(mean, std, size_max):
    dims = torch.clamp(((torch.randn(3) * std) + mean), 0.01, size_max).tolist()
    return [round(x, 3) for x in dims]
    # return [random.uniform(0.05, 0.5) for _ in range(3)]

def getFacePos(face):
    ft = {
        'right': ([1.0, 0.5, 0.5], 0, 0.),
        'left': ([0.0, 0.5, 0.5], 0, 1.),
        'top': ([.5, 1.0, 0.5], 1, 0.),
        'bot': ([.5, 0.0, 0.5], 1, 1.),
        'front': ([.5, 0.5, 1.0], 2, 0.),
        'back': ([.5, 0.5, 0.0], 2, 1.),
    }
    return ft[face]

def make_attach(name1, name2, face1, face2):
    _, ind, val = getFacePos(face1)
    val = 1-val
    # pick random point on face1
    attach1 = [round(random.random(), 3) for _ in range(2)]
    attach1 = attach1[:ind] + [val] + attach1[ind:]
    _, ind, val = getFacePos(face2)
    val = 1-val
    # pick random point on face2
    attach2 = [round(random.random(), 3) for _ in range(2)]
    attach2 = attach2[:ind] + [val] + attach2[ind:]
    x, y, z, u, v, w = attach1 + attach2
    return f"attach({name1}, {name2}, {x}, {y}, {z}, {u}, {v}, {w})"

def make_cuboid(name, mean, std, size_max, align_prob):
    dims = samp_dims(mean, std, size_max)
    a, b, c = dims
    return f"{name} = Cuboid({a}, {b}, {c}, {random.random() < align_prob})"

def inside_bbox(P):
    cuboid_objects = [v for k, v in P.cuboids.items() if k != 'bbox']
    bbox_corners = P.cuboids["bbox"].getCorners()
    max_x = torch.max(bbox_corners[:, 0])
    min_x = torch.min(bbox_corners[:, 0])
    max_y = torch.max(bbox_corners[:, 1])
    min_y = torch.min(bbox_corners[:, 1])
    max_z = torch.max(bbox_corners[:, 2])
    min_z = torch.min(bbox_corners[:, 2])
    corners = []
    for c in cuboid_objects:
        corners.append(c.getCorners())
    corners = torch.stack(corners)
    good_corners_x = (corners[:, :, 0] <= max_x).all() and (corners[:, :, 0] >= min_x).all()
    good_corners_y = (corners[:, :, 1] <= max_y).all() and (corners[:, :, 1] >= min_y).all()
    good_corners_z = (corners[:, :, 2] <= max_z).all() and (corners[:, :, 2] >= min_z).all()
    return good_corners_x and good_corners_y and good_corners_z

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
    good_corners = inside_bbox(P)
    # if not good_overlap:
    #     print("TOO MUCH OVERLAP")
    # if not good_corners:
    #     print("OUTSIDE BOX")
    return good_overlap and good_corners

def canonical(lines):
    P = Program()
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
    lines.sort(key = order)
    return lines

def rand_program(prog, max_cuboids, bbox_dims, hier_index):
    size_max = max(bbox_dims)
    num_cuboids = int(samp((max_cuboids / 3), 1, 2, max_cuboids))
    num_source = int(samp(1.5, 1, 1, min(max_sources, num_cuboids)))
    cdim_mean = np.mean(bbox_dims) / 2
    cdim_std = np.std(bbox_dims)
    leaf_prob = 0.5 ** (hier_index + 1)

    print(f"NUM SOURCE: {num_source}")
    print(f"NUM CUBOIDS: {num_cuboids}")

    def attach_source(name, P):
        counter = 0
        while counter < 100:
            old_prog = deepcopy(P)
            cuboid_line = make_cuboid(name, cdim_mean, cdim_std, size_max, align_prob)
            # bottom of bbox and bottom of (unattached) source cuboid
            attach_line = make_attach(name, "bbox", "bot", "bot")
            lines = [cuboid_line, attach_line]
            P.execute(cuboid_line)
            P.execute(attach_line)
            if valid(P):
                break
            P = old_prog
            counter += 1
        if counter >= 100:
            print("GAVE UP ON SOURCE")
            return P, []
        return P, lines

    def create_edge(src_cuboid, cuboids, P, new_src_cuboids, prev_attach, new_cuboid):
        lines = []
        faces = ['right', 'left', 'top', 'bot', 'front', 'back']

        new_attach = None
        possible_attachments = []
        for c in cuboids:
            # don't attach to ancestors, sources, or cuboids already attached to in this cycle
            if not (c["id"] in src_cuboid["ancestors"] or c["level"] == 1 or c["id"] in prev_attach):
                if c['name'] == 'bbox' and src_cuboid['level'] == 1:
                    continue
                possible_attachments.append(c)
        if new_cuboid or possible_attachments == []:
            if len(P.cuboids) >= num_cuboids:
                print("TOO MANY CUBOIDS")
                return P, new_src_cuboids, [], new_attach
            if new_src_cuboids == []:
                index = max([x["id"] for x in cuboids]) + 1
            else:
                index = max([x["id"] for x in new_src_cuboids]) + 1
            new_src_cuboids.append({"name": f"cube{index}",
                                    "id": index,
                                    "ancestors": src_cuboid["ancestors"] | set([index]),
                                    "level": src_cuboid["level"] + 1})
            cuboid_line = make_cuboid(f"cube{index}", cdim_mean, cdim_std, size_max, align_prob)
            lines = [cuboid_line]
            P.execute(cuboid_line)
            c_new = f"cube{index}"

            face1 = faces[random.choice(range(6))]
            face2 = faces[random.choice(range(6))]
            attach_line = make_attach(c_new, src_cuboid["name"], face1, face2)
        else:
            # pos_levels = [c['level'] for c in possible_attachments]
            # p = [-1*(x - max(pos_levels))+1.0 for x in pos_levels]
            # p = p / np.sum(p)
            selected_cuboid = np.random.choice(possible_attachments)
            # selected_cuboid['ancestors'] = selected_cuboid['ancestors'] | src_cuboid['ancestors']
            src_cuboid['ancestors'] = selected_cuboid['ancestors'] | src_cuboid['ancestors']
            c_new = selected_cuboid["name"]
            new_attach = selected_cuboid["id"]

            if not c_new == "bbox":
                face1 = faces[random.choice(range(6))]
                face2 = faces[random.choice(range(6))]
                attach_line = make_attach(src_cuboid["name"], c_new, face1, face2)
            # if bbox only attach to top face and reverse direction of edge
            else:
                attach_line = make_attach(src_cuboid["name"], c_new, "top", "top")

        lines.append(attach_line)
        P.execute(attach_line)

        return P, new_src_cuboids, lines, new_attach

    def add_attachment(src_cuboid, cuboids, P, num_sources):
        p = np.full((max_out + 1,), 1.0)
        p[0] = p[0] / 5
        p[3] = p[3] / 5
        p[2] = p[2] / 2
        p = p / np.sum(p)
        out = np.random.choice(list(range(max_out+1)), p=p)
        print(f"OUT: {out}")

        new_src_cuboids = []
        lines = []
        prev_attach = []

        for edge in range(out):
            edge_counter = 0
            new_cuboid = random.random() < new_cuboid_prob
            if len(P.cuboids) == num_cuboids:
                new_cuboid = False
            while edge_counter < 100:
                old_prog = deepcopy(P)
                old_new_src_cuboids = deepcopy(new_src_cuboids)

                P, new_src_cuboids, new_lines, new_attach = create_edge(src_cuboid, cuboids, \
                                                                                 P, new_src_cuboids, prev_attach, new_cuboid)
                if valid(P):
                    if not new_cuboid:
                        print("MADE ATTACHMENT WITH PREV CUBOID")
                    else:
                        print("MADE NEW CUBOID")
                    prev_attach.append(new_attach)
                    lines += new_lines
                    break

                P = old_prog
                new_src_cuboids = old_new_src_cuboids
                edge_counter += 1

            if not out == 0 and edge_counter >= 100:
                print("GAVE UP ON EDGE")

        return new_src_cuboids, P, lines

    P = Program()
    bbox_line = f"bbox = Cuboid({bbox_dims[0]}, {bbox_dims[1]}, {bbox_dims[2]}, True)"
    P.execute(bbox_line)
    lines = [bbox_line]
    current_cuboids = []
    src_count = 0
    for _ in range(num_source):
        P, new_lines = attach_source(f"cube{src_count}", P)
        if len(new_lines) > 0:
            lines += new_lines
            current_cuboids.append({"name": f"cube{src_count}",
                                    "id": src_count,
                                    "ancestors": set([src_count]),
                                    "level": 1})
            src_count += 1
    if len(current_cuboids) <= 1:
        print("COULDNT FIT MORE THAN ONE CUBOID")
        return None
    cuboids = [{"name": "bbox", "id": -1, "ancestors": set([-1]), "level": 0}] + current_cuboids
    while len(current_cuboids) > 0:
        c = current_cuboids.pop(0)
        new_cuboids, P, new_lines = add_attachment(c, cuboids, P, num_source)
        lines += new_lines
        current_cuboids += new_cuboids
        cuboids += new_cuboids

    # choose from the largest cuboids to expand
    non_bbox_cuboids = [x for x in P.cuboids if not x == "bbox" and torch.prod(P.cuboids[x].dims) > 0.02]
    sorted_cuboids = sorted(non_bbox_cuboids, key = lambda x: -torch.prod(P.cuboids[x].dims))
    # num_sub = len([_ for _ in range(len(non_bbox_cuboids)) if random.random() < leaf_prob])
    num_sub = len(sorted_cuboids)
    sub_cuboids = sorted_cuboids[:num_sub]
    print(sub_cuboids)

    prog['prog'] = canonical(lines)
    next_q = []
    children = []
    for c in P.cuboids:
        if c in sub_cuboids:
            cprog = {"prog": None, "children": None}
            children.append(cprog)
            next_q.append((cprog, hier_index+1, P.cuboids[c].dims.tolist()))
        else:
            children.append({})
    prog['children'] = children

    return next_q

def rand_hier_program():
    q = []
    hier_prog = {"prog": None, "children": None}
    hier_index = 0
    num_cuboids = 20
    bbox_dims = samp_dims(1, .1, 2)
    next_q = rand_program(hier_prog, num_cuboids, bbox_dims, hier_index)
    while next_q is None:
        bbox_dims = samp_dims(1, .1, 2)
        next_q = rand_program(hier_prog, num_cuboids, bbox_dims, hier_index)
    num_prev_children = len(hier_prog['children']) - 1
    q += next_q
    while not q == []:
        num_cuboids = num_cuboids - num_prev_children
        prog, hier_index, bbox_dims = q.pop(0)
        next_q = rand_program(prog, num_cuboids, bbox_dims, hier_index)
        if next_q is None:
            prog.pop('prog')
            prog.pop('children')
            num_prev_children = 0
        else:
            q += next_q
            num_prev_children = len(prog['children']) - 1
    return hier_prog

for n in range(10):
    prog = rand_hier_program()
    verts, faces = hier_execute(prog)
    writeObj(verts, faces, f'{n}.obj')
    writeHierProg(prog, f"{n}.txt")
