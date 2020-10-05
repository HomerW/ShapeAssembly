import torch
from ShapeAssembly import hier_execute, Program
import utils

device = torch.device("cuda")

def prog_to_pc(prog, ns):
    if prog['prog'] == []:
        return torch.zeros((ns, 3))
    verts, faces = hier_execute(prog)
    for i in range(3):
        verts[:,i] = verts[:,i] - verts[:,i].mean()
    pc = utils.sample_surface(faces, verts.unsqueeze(0), ns, return_normals=False)[0]
    return pc

def get_line_pc(full_pc, partial_prog, ns):
    full_labels = torch.zeros((ns // 2, 1))
    full_labeled = torch.cat([full_pc, full_labels], 1)

    pverts, pfaces = partial_prog.getShapeGeo()
    if pverts == None:
        partial_pc = torch.zeros((ns // 2, 3))
    else:
        for i in range(3):
            pverts[:,i] = pverts[:,i] - pverts[:,i].mean()
        partial_pc = utils.sample_surface(pfaces, pverts.unsqueeze(0), ns // 2, return_normals=False)[0]
    partial_labels = torch.ones((ns // 2, 1))
    partial_labeled = torch.cat([partial_pc, partial_labels], 1)

    pc = torch.cat([full_labeled, partial_labeled], 0)
    return pc.transpose(1, 0).unsqueeze(0)

def get_train_pc(prog, ns):
    full_pc = prog_to_pc(prog, ns // 2)
    partial_prog = Program()
    pc_list = [get_line_pc(full_pc, partial_prog, ns)]
    for i in range(len(prog['prog'])):
        partial_prog.execute(prog['prog'][i])
        pc_list.append(get_line_pc(full_pc, partial_prog, ns))
    train_pc = torch.cat(pc_list, 0)
    return train_pc

def reencode_prog(full_pc, partial_prog, ns, encoder):
    full_labeled = full_pc.transpose(1, 0)
    pverts, pfaces = partial_prog.getShapeGeo()
    if pverts == None:
        partial_pc = torch.zeros((ns // 2, 3))
    else:
        for i in range(3):
            pverts[:,i] = pverts[:,i] - pverts[:,i].mean()
        partial_pc = utils.sample_surface(pfaces, pverts.unsqueeze(0), ns // 2, return_normals=False)[0]
    partial_labels = torch.ones((ns // 2, 1))
    partial_labeled = torch.cat([partial_pc, partial_labels], 1).to(device)

    pc = torch.cat([full_labeled, partial_labeled], 0)
    pc = pc.transpose(1, 0).unsqueeze(0)
    return encoder(pc).unsqueeze(0)
