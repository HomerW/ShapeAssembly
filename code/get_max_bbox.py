from ShapeAssembly import Program
import os
import torch

P = Program()

files = os.listdir("data/chair")

dim_list = []
src_avgs = []
for f in files:
    with open(f"data/chair/{f}", "r") as file:
        lines = file.readlines()
    src_count = 0
    seen_cuboids = set()
    not_child = True
    for l in lines:
        # if "Assembly Program_1" in l:
        #     not_child = False
        # if not_child:
        #     continue
        # if "Cuboid" in l and ("bbox" in l):
        #     src_count += 1
        #     parse = P.parseCuboid(l)
        #     dim_list += parse[1:4]
        if "attach" in l and "bbox" in l:
            parse = P.parseAttach(l)
            # dim_list += [float(parse[-1])]
            if parse[0] not in seen_cuboids:
                src_count += 1
                seen_cuboids.add(parse[0])
        if "squeeze" in l and "bbox" in l:
            parse = P.parseSqueeze(l)
            if parse[0] not in seen_cuboids:
                src_count += 1
                seen_cuboids.add(parse[0])
        if "Assembly Program_1" in l:
            break
            # mean = torch.mean(torch.stack(parse[1:4])).item()
            # overall_mean += mean
            # num_lines += 1
            # if max_dim > overall_max_dim:
            #     overall_max_dim = max_dim
    # if not not_child:
    print(float(src_count))
    src_avgs.append(float(src_count))

dim_tensor = torch.tensor(dim_list)
print(dim_tensor)
print(torch.mean(dim_tensor))
print(torch.std(dim_tensor))
