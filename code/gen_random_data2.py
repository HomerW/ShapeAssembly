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

align_prob = 0.8
max_sources = 4
max_out = 3
max_att_overlap = 0.15
max_disc_overlap = 0.05
squeeze_prob = 0.3
kill_prob = 0.2
reflect_prob = 0.7

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

def make_squeeze(name1, name3, face):
    # pick random point
    u, v = [round(random.random(), 2) for _ in range(2)]
    return f"squeeze({name1}, bbox, {name3}, {face}, {u}, {v})"

def make_cuboid(name, mean, std, size_max, aligned):
    dims = samp_dims(mean, std, size_max)
    a, b, c = dims
    return f"{name} = Cuboid({a}, {b}, {c}, {aligned})", dims

def make_reflect(name):
    axes = ["X", "Y", "Z"]
    axis = np.random.choice(axes, p=[0.8, 0.1, 0.1])
    return f"reflect({name}, {axis})"

def make_translate(name):
    axes = ["X", "Y", "Z"]
    axis = np.random.choice(axes, p=[0.7, 0.15, 0.15])
    num = int(samp(4, 2, 1, 10))
    return f"translate({name}, {axis}, {num}, {random.random()})"

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
    num_cuboids = int(samp((max_cuboids / 1.5), 1, 3, max_cuboids))
    num_source = int(samp(1.75, 0.5, 1, min(max_sources, num_cuboids)))
    cdim_mean = np.mean(bbox_dims) / 2
    cdim_std = cdim_mean / 2.5
    leaf_prob = 0.5 ** (hier_index + 1)

    print(f"NUM SOURCE: {num_source}")
    print(f"NUM CUBOIDS: {num_cuboids}")

    def attach_source(name, P):
        counter = 0
        while counter < 100:
            old_prog = deepcopy(P)
            aligned = random.random() < align_prob
            cuboid_line, dims = make_cuboid(name, cdim_mean, cdim_std, size_max, aligned)
            # check if sqeezing into bbox is a good fit
            good_squeeze = ((dims[1] / bbox_dims[1]) > 0.9)

            if random.random() < squeeze_prob and (not aligned or good_squeeze):
                attach_line = make_squeeze(name, "bbox", "bot")
            else:
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
            return P, [], None
        return P, lines, dims

    def extend_cuboid(src_cuboid, cuboids, P):
        faces = ['right', 'left', 'top', 'bot', 'front', 'back']
        lines = []
        new_cuboids = []
        next_q = []

        num_new_cuboids = int(samp(1, 0.5, 1, max_out))

        for _ in range(num_new_cuboids):
            if new_cuboids == []:
                index = max([x["id"] for x in cuboids]) + 1
            else:
                index = max([x["id"] for x in new_cuboids]) + 1
            name = f"cube{index}"

            # make initial attachment to src cuboid
            if (len(cuboids) + len(new_cuboids)) < (num_cuboids - 1):
                counter = 0
                while counter < 100:
                    old_prog = deepcopy(P)
                    old_new_cuboids = deepcopy(new_cuboids)
                    old_next_q = deepcopy(next_q)

                    aligned = random.random() < align_prob
                    cuboid_line, dims = make_cuboid(name, cdim_mean, cdim_std, size_max, aligned)
                    nc = {"name": f"cube{index}",
                          "id": index,
                          "ancestor": src_cuboid["name"],
                          "dims": dims}
                    new_cuboids.append(nc)

                    attach_lines = []
                    if random.random() < squeeze_prob:
                        # faces = ['top', 'bot']
                        # face = random.choice(faces)
                        attach_line = make_squeeze(name, src_cuboid["name"], 'top')
                    else:
                        next_q.append(nc)
                        attach_line = make_attach(name, src_cuboid["name"], random.choice(faces), random.choice(faces))

                    P.execute(cuboid_line)
                    P.execute(attach_line)
                    if valid(P):
                        lines += [cuboid_line, attach_line]
                        break
                    P = old_prog
                    new_cuboids = old_new_cuboids
                    next_q = old_next_q
                    counter += 1
                if counter >= 100:
                    print("GAVE UP ON EXTENSION")
                    return [], [], P, []

                # if cuboid is not aligned potentially add some more attachments
                counter = 0
                while counter < 100 and not aligned:
                    old_prog = deepcopy(P)
                    num_extra_attaches = int(samp(1, 0.5, 0, 3))
                    print(f"EXTRA ATTACHES {num_extra_attaches}")

                    attach_lines = []
                    for _ in range(num_extra_attaches):
                        possible_cuboids = [c["name"] for c in cuboids if not c["name"] == c["ancestor"]]
                        attach_cuboid = random.choice(possible_cuboids)
                        attach_lines.append(make_attach(name, attach_cuboid, random.choice(faces), random.choice(faces)))
                    for l in attach_lines:
                        P.execute(l)
                    if valid(P):
                        lines += attach_lines
                        break
                    P = old_prog
                    counter += 1
                    if counter >= 100:
                        print("GAVE UP ON EXTENSION II")

        return new_cuboids, next_q, P, lines


    P = Program()
    bbox_line = f"bbox = Cuboid({bbox_dims[0]}, {bbox_dims[1]}, {bbox_dims[2]}, True)"
    print(bbox_line)
    P.execute(bbox_line)
    lines = [bbox_line]
    q = []
    src_count = 0
    for _ in range(num_source):
        P, new_lines, dims = attach_source(f"cube{src_count}", P)
        if len(new_lines) > 0:
            lines += new_lines
            for l in new_lines:
                print(l)
            q.append({"name": f"cube{src_count}",
                      "id": src_count,
                      "ancestor": "bbox",
                      "dims": dims})
            src_count += 1
    if len(q) == 0:
        print("COULDNT FIT ANY CUBOIDS")
        return None
    cuboids = deepcopy(q)
    while len(q) > 0 and len(cuboids) < (num_cuboids - 1):
        c = q.pop(0)
        new_cuboids, next_q, P, new_lines = extend_cuboid(c, cuboids, P)
        lines += new_lines
        for l in new_lines:
            print(l)
        q += next_q
        cuboids += new_cuboids

    # add some symmetry macros
    num_sym = int(samp(1, 0.5, 0, 3))
    for _ in range(num_sym):
        counter = 0
        while counter < 100:
            old_prog = deepcopy(P)
            sym_cuboid = random.choice([c["name"] for c in cuboids])
            if random.random() < reflect_prob:
                new_line = make_reflect(sym_cuboid)
            else:
                new_line = make_translate(sym_cuboid)
            P.execute(new_line)
            if valid(P):
                lines.append(new_line)
                print(new_line)
                break
            P = old_prog
            counter += 1
            if counter >= 100:
                print("GAVE UP ON MACRO")

    # correct dimensions since cuboids might have been scaled during execution
    for c in cuboids:
        new_dims = [round(x, 3) for x in P.cuboids[c['name']].dims.tolist()]
        c['dims'] = new_dims
        new_lines = []
        for l in lines:
            if (c['name'] in l) and ("Cuboid" in l):
                aligned = P.parseCuboid(l)[-1]
                new_lines.append(f"{c['name']} = Cuboid({new_dims[0]}, {new_dims[1]}, {new_dims[2]}, {aligned})")
            else:
                new_lines.append(l)
        lines = new_lines

    # choose from the largest cuboids to expand
    non_bbox_cuboids = [x for x in cuboids if not x['name'] == "bbox" and np.prod(x["dims"]) > 0.02]
    sorted_cuboids = sorted(non_bbox_cuboids, key = lambda x: -np.prod(x["dims"]))
    num_sub = len([_ for _ in range(len(non_bbox_cuboids)) if random.random() < leaf_prob])
    # num_sub = len(sorted_cuboids)
    sub_cuboids = sorted_cuboids[:num_sub]

    # prog['prog'] = canonical(lines)
    prog['prog'] = lines
    next_q = []
    # start of with bbox child
    children = [{}]
    for c in cuboids:
        if c in sub_cuboids:
            cprog = {"prog": None, "children": None}
            children.append(cprog)
            next_q.append((cprog, hier_index+1, c["dims"]))
        else:
            children.append({})
    prog['children'] = children

    return next_q

def rand_hier_program():
    q = []
    hier_prog = {"prog": None, "children": None}
    hier_index = 0
    num_cuboids = 11
    bbox_dims = samp_dims(1, .1, 2)
    next_q = rand_program(hier_prog, num_cuboids, bbox_dims, hier_index)
    while next_q is None:
        bbox_dims = samp_dims(1, .1, 2)
        next_q = rand_program(hier_prog, num_cuboids, bbox_dims, hier_index)
    num_prev_children = len(hier_prog['children'])
    num_cuboids = num_cuboids - num_prev_children + 2
    q += next_q
    while len(q) > 0:
        prog, hier_index, bbox_dims = q.pop(0)
        next_q = rand_program(prog, num_cuboids, bbox_dims, hier_index)
        if next_q is None:
            prog.pop('prog')
            prog.pop('children')
            num_prev_children = 0
        else:
            q += next_q
            num_prev_children = len(prog['children'])
        num_cuboids = num_cuboids - num_prev_children + 2

        # if not enough remaining cuboids to keep expanding
        # pull all children out of q and make them leaves
        if not num_cuboids > 2:
            while len(q) > 0:
                prog, hier_index, bbox_dims = q.pop(0)
                prog.pop('prog')
                prog.pop('children')
            break

    return hier_prog

for n in range(10):
    prog = rand_hier_program()
    verts, faces = hier_execute(prog)
    writeObj(verts, faces, f'{n}.obj')
    writeHierProg(prog, f"{n}.txt")
