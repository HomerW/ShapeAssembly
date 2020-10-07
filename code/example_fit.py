import sys
import torch
from utils import sample_surface, writePC, writeHierProg
from losses import ChamferLoss
from ShapeAssembly import ShapeAssembly, writeObj, make_hier_prog
import os
import metrics

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


def fit(prog_path, obj_path, out_path):
    progs = os.listdir(prog_path)
    objs = os.listdir(obj_path)
    fitted_progs = []
    for prg in progs:
        sa = ShapeAssembly()
        p_no_e = prg.split("_")[1]
        index = int(p_no_e.split(".")[0])

        # should be shape N x 3
        target_pc = load_point_cloud(f"{obj_path}/{index}.obj")

        out_file = f"{out_path}/{index}"
        with open(f"{prog_path}/{prg}") as file:
            lines = file.readlines()
        hier, param_dict, param_list = sa.make_hier_param_dict(lines)

        opt = torch.optim.Adam(param_list, 0.001)

        start = torch.cat(param_list).clone()

        for iter in range(400):
            print(iter)
            verts, faces = sa.diff_run(hier, param_dict)

            samps = sample_surface(faces, verts.unsqueeze(0), 10000)
            closs = cham_loss(
                samps.squeeze().T.unsqueeze(0).cuda(),
                target_pc.T.unsqueeze(0).cuda(),
                0.0
            )

            ploss = (torch.cat(param_list) - start).abs().sum()

            loss = closs + ploss.cuda() * 0.001

            opt.zero_grad()
            loss.backward()
            opt.step()

        writeObj(verts, faces, out_file + '.obj')
        sa.fill_hier(hier, param_dict)
        writeHierProg(hier, out_file + '.txt')

        fitted_progs.append((hier, index))

    return fitted_progs

if __name__ == '__main__':
    progs_path = sys.argv[1]
    gt_progs_path = sys.argv[2]
    progs = os.listdir(progs_path)
    gt_prog_dicts = {}

    recon_sets = []
    for p in progs[:10]:
        sa = ShapeAssembly()
        p_no_e = p.split("_")[1]
        index = int(p_no_e.split(".")[0])
        lines = sa.load_lines(f"{progs_path}/{p}")
        gt_lines = sa.load_lines(f"{gt_progs_path}/{p_no_e}")
        prog = make_hier_prog(lines)
        gt_prog = make_hier_prog(gt_lines)
        gt_prog_dicts[index] = gt_prog
        recon_sets.append((prog, gt_prog, index))


    recon_results, _ = metrics.recon_metrics(
        recon_sets, "program_fit", "fit", "generated", 0, True
    )

    print(recon_results)

    fitted_progs = fit("program_fit/fit/programs/generated", "program_fit/fit/objs/gt", "program_fit/fitted")

    # fitted_progs = []
    # fitted_prog_list = os.listdir("program_fit/fitted")
    # for p in fitted_prog_list:
    #     if p[-3:] == "txt":
    #         index = int(p.split(".")[0])
    #         lines = sa.load_lines(f"program_fit/fitted/{p}")
    #         prog = make_hier_prog(lines)
    #         fitted_progs.append((prog, index))

    fitted_recon_sets = [(fitted_prog, gt_prog_dicts[index], index) for (fitted_prog, index) in fitted_progs]

    fitted_recon_results, _ = metrics.recon_metrics(
        fitted_recon_sets, "program_fit", "fitted", "generated", 0, True
    )

    print(fitted_recon_results)
