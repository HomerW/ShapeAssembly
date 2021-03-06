import sys
import torch
from utils import sample_surface, writePC, writeHierProg, loadObj
from losses import ChamferLoss
from ShapeAssembly import ShapeAssembly, writeObj
import time

device = torch.device("cuda")
cham_loss = ChamferLoss(device)

def create_point_cloud(in_file, out_file):
    sa = ShapeAssembly()
    lines = sa.load_lines(sys.argv[1])
    hier, param_dict, _ = sa.make_hier_param_dict(lines)
    verts, faces = sa.diff_run(hier, param_dict)
    tsamps = sample_surface(faces, verts.unsqueeze(0), 10000).squeeze()
    writePC(tsamps, out_file)

def load_point_cloud(pc_file):
    pc = []
    with open(pc_file) as f:
        for line in f:
            ls = line.split()
            if len(ls) == 0:
                continue
            if ls[0] == 'v':
                pc.append([
                    float(ls[1]),
                    float(ls[2]),
                    float(ls[3]),
                    0.0,
                    0.0,
                    0.0
                ])
    return torch.tensor(pc)


def main():
    sa = ShapeAssembly()
    lines = sa.load_lines(sys.argv[1])

    # should be shape N x 3
    tverts, tfaces = loadObj(sys.argv[2])

    tverts = torch.tensor(tverts)
    tfaces = torch.tensor(tfaces).long()
    tsamps = sample_surface(tfaces, tverts.unsqueeze(0), 10000)

    out_file = sys.argv[3]
    hier, param_dict, param_list = sa.make_hier_param_dict(lines)

    start_time = time.time()
    opt = torch.optim.Adam(param_list, 0.001)

    start = torch.cat(param_list).clone()

    for iter in range(400):
        verts, faces = sa.diff_run(hier, param_dict)

        samps = sample_surface(faces, verts.unsqueeze(0), 10000)
        # tsamps = sample_surface(tfaces, tverts.unsqueeze(0), 10000)
        closs = cham_loss(
            samps.squeeze().T.unsqueeze(0).cuda(),
            tsamps.squeeze().T.unsqueeze(0).cuda(),
            0.0
        )

        ploss = (torch.cat(param_list) - start).abs().sum()

        loss = closs + ploss.cuda() * 0.001
        print(float(loss))

        opt.zero_grad()
        loss.backward()
        opt.step()

        # if iter % 10 == 0:
        #     writeObj(verts, faces, f'{iter}_' + out_file + '.obj')

    end_time = time.time()
    print(f"TIME: {end_time-start_time}")

    writeObj(verts, faces, out_file + '.obj')
    sa.fill_hier(hier, param_dict)
    writeHierProg(hier, out_file + '.txt')

if __name__ == '__main__':
    main()
