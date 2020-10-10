from torch.utils.data import DataLoader
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import losses
from parse_prog import progToTarget, predToProg
from argparse import Namespace
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import argparse
from tqdm import tqdm
import ast
import metrics
from pointnet_fixed_reencode import PointNetEncoder as PCEncoder
# from pc_encoder import PCEncoder
from model_prog_reencode import FDGRU, load_progs, progToData, getInds, _col, \
                                run_train_decoder, run_eval_decoder, getLossConfig
from ShapeAssembly import hier_execute, Program
from reencode import get_train_pc

num_samps = 10000
hidden_dim = 256
batch_size = 1
num_eval = 50
ADAM_EPS = 1e-6
dec_lr = 0.0001
enc_lr = 0.0001
dec_step = 5000
dec_decay = 1.0
enc_step = 5000
enc_decay = 1.0
device = torch.device("cuda")

# def prog_to_pc(prog, ns):
#     verts, faces = hier_execute(prog)
#     for i in range(3):
#         verts[:,i] = verts[:,i] - verts[:,i].mean()
#     pc = utils.sample_surface(faces, verts.unsqueeze(0), ns, return_normals=False)[0]
#     return pc

# just takes root program
def lines_to_pc(lines):
    P = Program()
    for i, line in enumerate(lines):
        P.execute(line)
    verts, faces = P.getShapeGeo()
    for i in range(3):
        verts[:,i] = verts[:,i] - verts[:,i].mean()
    pc = utils.sample_surface(faces, verts.unsqueeze(0), num_samps, return_normals=False)[0]
    return pc

def lines_to_prog(lines):
    P = Program()
    num_children = len([l for l in lines if "Cuboid(" in l])
    def order(l):
        if "Cuboid(" in l:
            name = P.parseCuboid(l)[0]
            if name == "bbox":
                return 0
            else:
                return int(name[4:]) + 1
        else:
            return 100
    lines.sort(key = order)
    prog = {
        "name": 0,
        "prog": lines,
        "children": [{}] * num_children
    }
    return prog

# Losses that will be used to train the model
# def getLossConfig():
#     loss_config = {
#         'cmd': 1.,
#         'cub_prm': 50.,
#
#         'xyz_prm': 50.,
#         'uv_prm': 50.,
#         'sym_prm': 50.,
#
#         'cub': 1.,
#         'sym_cub': 1.,
#         'sq_cub': 1.,
#
#         'leaf': 1.,
#         'bb': 50.,
#
#         'axis': 1.,
#         'face': 1.,
#         'align': 1.
#     }
#
#     return loss_config

def prog_to_gpu(prog):
    if len(prog) == 0:
        return prog
    prog["inp"] = prog["inp"].to(device)
    prog["tar"] = prog["tar"].to(device)
    prog["weights"] = prog["weights"].to(device)
    prog["bb_dims"] = prog["bb_dims"].to(device)
    for c in prog["children"]:
        prog_to_gpu(c)
    return prog

def get_partnet_data(dataset_path, category, max_shapes):
    class PartNetDataset(torch.utils.data.Dataset):
        def __init__(self, progs):
            self.progs = progs
            self.pcs = [get_train_pc(prog, num_samps).cpu() for (ind, prog) in self.progs]
            self.prog_data = [progToData(prog) for (ind, prog) in self.progs]

        def __getitem__(self, idx):
            ind, prog = self.progs[idx]
            prog_pc = self.pcs[idx].to(device)
            nprog = prog_to_gpu(self.prog_data[idx])
            return (nprog, prog_pc, ind)

        def __len__(self):
            return len(self.progs)

    raw_indices, progs = load_progs(dataset_path, max_shapes)

    inds_and_progs = list(zip(raw_indices, progs))
    random.shuffle(inds_and_progs)

    inds_and_progs = inds_and_progs[:max_shapes]

    train_ind_file = f'data_splits/{category}/train.txt'
    val_ind_file = f'data_splits/{category}/val.txt'
    train_inds = getInds(train_ind_file)
    val_inds = getInds(val_ind_file)
    train_progs = [(ind, prog) for (ind, prog) in inds_and_progs if ind in train_inds]
    val_progs = [(ind, prog) for (ind, prog) in inds_and_progs if ind in val_inds]

    train_num = len(train_progs)
    val_num = len(val_progs)
    print(f"Training size: {train_num}")
    print(f"Validation size: {val_num}")

    train_dataset = DataLoader(PartNetDataset(train_progs), batch_size, shuffle=True, collate_fn = _col)
    eval_train_dataset = DataLoader(PartNetDataset(train_progs[:num_eval]), batch_size=1, shuffle=False, collate_fn = _col)
    val_dataset = DataLoader(PartNetDataset(val_progs), batch_size, shuffle = False, collate_fn = _col)
    eval_val_dataset = DataLoader(PartNetDataset(val_progs[:num_eval]), batch_size=1, shuffle = False, collate_fn = _col)

    return train_dataset, val_dataset, eval_train_dataset, eval_val_dataset

def get_random_data(datapath, max_shapes):
    class WakeSleepDataset():
        def __init__(self, datapath, max_shapes):
            super(WakeSleepDataset, self).__init__()
            self.data = []
            for i in range(min(len(os.listdir(datapath)), max_shapes)):
                with open(f"{datapath}/{i}.txt") as file:
                    lines = file.readlines()
                prog = lines_to_prog(lines)
                pc = lines_to_pc(lines)
                nprog = progToData(prog)
                self.data.append((nprog, pc, i))
            self.data = self.data[:max_shapes]

        def __len__(self):
            return len(self.data)

        def __getitem__(self,index):
            return self.data[index]

    dataset = WakeSleepDataset(datapath, max_shapes)
    loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle = True,
        collate_fn = _col
    )

    return loader

def print_eval_results(eval_results, name):
        if eval_results['nc'] > 0:
            eval_results['cub_prm'] /= eval_results['nc']

        if eval_results['na'] > 0:
            eval_results['xyz_prm'] /= eval_results['na']
            eval_results['cubc'] /= eval_results['na']

        if eval_results['count'] > 0:
            eval_results['bb'] /= eval_results['count']

        if eval_results['nl'] > 0:
            eval_results['cmdc'] /= eval_results['nl']

        if eval_results['ns'] > 0:
            eval_results['sym_cubc'] /= eval_results['ns']
            eval_results['axisc'] /= eval_results['ns']

        if eval_results['nsq'] > 0:
            eval_results['uv_prm'] /= eval_results['nsq']
            eval_results['sq_cubc'] /= eval_results['nsq']
            eval_results['facec'] /= eval_results['nsq']

        if eval_results['nap'] > 0:
            eval_results['palignc'] /= eval_results['nap']

        if eval_results['nan'] > 0:
            eval_results['nalignc'] /= eval_results['nan']

        eval_results.pop('nc')
        eval_results.pop('nan')
        eval_results.pop('nap')
        eval_results.pop('na')
        eval_results.pop('ns')
        eval_results.pop('nsq')
        eval_results.pop('nl')
        eval_results.pop('count')
        eval_results.pop('cub')
        eval_results.pop('sym_cub')
        eval_results.pop('axis')
        eval_results.pop('cmd')
        eval_results.pop('miss_hier_prog')

        print(
f"""

Evaluation on {name} set:

Eval {name} F-score = {eval_results['fscores']}
Eval {name} IoU = {eval_results['iou_shape']}
Eval {name} PD = {eval_results['param_dist_parts']}
Eval {name} Prog Creation Perc: {eval_results['prog_creation_perc']}
Eval {name} Cub Prm Loss = {eval_results['cub_prm']}
Eval {name} XYZ Prm Loss = {eval_results['xyz_prm']}
Eval {name} UV Prm Loss = {eval_results['uv_prm']}
Eval {name} Sym Prm Loss = {eval_results['sym_prm']}
Eval {name} BBox Loss = {eval_results['bb']}
Eval {name} Cmd Corr % {eval_results['cmdc']}
Eval {name} Cub Corr % {eval_results['cubc']}
Eval {name} Squeeze Cub Corr % {eval_results['sq_cubc']}
Eval {name} Face Corr % {eval_results['facec']}
Eval {name} Pos Align Corr % {eval_results['palignc']}
Eval {name} Neg Align Corr % {eval_results['nalignc']}
Eval {name} Sym Cub Corr % {eval_results['sym_cubc']}
Eval {name} Sym Axis Corr % {eval_results['axisc']}
Eval {name} Corr Line # % {eval_results['corr_line_num']}
Eval {name} Bad Leaf % {eval_results['bad_leaf']}""")

def print_train_results(ep_result):
    arl = 0.
    loss_config = getLossConfig()

    for loss in loss_config:
        ep_result[loss] /= bc
        if loss == 'kl':
            continue
        if torch.is_tensor(ep_result[loss]):
            arl += ep_result[loss].detach().item()
        else:
            arl += ep_result[loss]

    ep_result['recon'] = arl
    if ep_result['nl'] > 0:
        ep_result['cmdc'] /= ep_result['nl']
    if ep_result['na'] > 0:
        ep_result['cubc'] /= ep_result['na']
    if ep_result['nap'] > 0:
        ep_result['palignc'] /= ep_result['nap']
    if ep_result['nan'] > 0:
        ep_result['nalignc'] /= ep_result['nan']
    if ep_result['ns'] > 0:
        ep_result['sym_cubc'] /= ep_result['ns']
        ep_result['axisc'] /= ep_result['ns']
    if ep_result['nsq'] > 0:
        ep_result['sq_cubc'] /= ep_result['nsq']
        ep_result['facec'] /= ep_result['nsq']

    ep_result.pop('na')
    ep_result.pop('nl')
    ep_result.pop('nc')
    ep_result.pop('nap')
    ep_result.pop('nan')
    ep_result.pop('ns')
    ep_result.pop('nsq')

    print(
        f"""
Recon Loss = {ep_result['recon']}
Cmd Loss = {ep_result['cmd']}
Cub Prm Loss = {ep_result['cub_prm']}
XYZ Prm Loss = {ep_result['xyz_prm']}
UV Prm Loss = {ep_result['uv_prm']}
Sym Prm Loss = {ep_result['sym_prm']}
Cub Loss = {ep_result['cub']}
Squeeze Cub Loss = {ep_result['sq_cub']}
Sym Cub Loss = {ep_result['sym_cub']}
Sym Axis Loss = {ep_result['axis']}
Face Loss = {ep_result['face']}
Align Loss = {ep_result['align']}
KL Loss = {ep_result['kl'] if 'kl' in ep_result else None}
BBox Loss = {ep_result['bb']}
Cmd Corr % {ep_result['cmdc']}
Cub Corr % {ep_result['cubc']}
Sq Cubb Corr % {ep_result['sq_cubc']}
Face Corr % {ep_result['facec']}
Align Pos Corr = {ep_result['palignc']}
Align Neg Corr = {ep_result['nalignc']}
Sym Cub Corr % {ep_result['sym_cubc']}
Sym Axis Corr % {ep_result['axisc']}""")

# Runs an epoch of evaluation logic
def model_eval(dataset, encoder, decoder, outpath, exp_name, epoch):
    decoder.eval()
    encoder.eval()

    named_results = {
        'count': 0.,
        'miss_hier_prog': 0.
    }

    recon_sets = []
    ep_result = {}

    for batch in dataset:
        shape, points, ind = batch[0]
        named_results[f'count'] += 1.

        prog, shape_result = run_eval_decoder(
            points, encoder, decoder, False, shape
        )

        # print(prog['prog'])

        for key in shape_result:
            nkey = f'{key}'
            if nkey not in named_results:
                named_results[nkey] = shape_result[key]
            else:
                named_results[nkey] += shape_result[key]

        if prog is None:
            named_results[f'miss_hier_prog'] += 1.
            continue

        recon_sets.append((prog, shape, ind))

    # For reconstruction, get metric performance
    recon_results, recon_misses = metrics.recon_metrics(
        recon_sets, outpath, exp_name, "train_dataset", epoch, True
    )

    # print(recon_results)

    for key in recon_results:
        named_results[key] = recon_results[key]

    named_results[f'miss_hier_prog'] += recon_misses

    named_results[f'prog_creation_perc'] = (
        named_results[f'count'] - named_results[f'miss_hier_prog']
    ) / named_results[f'count']

    return named_results

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

train_dataset, val_dataset, eval_train_dataset, eval_val_dataset = get_partnet_data("parse_flat_chair/reorder", "chair", 10000)
# train_dataset = get_random_data("random_data", 100)

encoder = PCEncoder()
encoder.to(device)
decoder = FDGRU(hidden_dim)
decoder.to(device)
dec_opt = torch.optim.Adam(
    decoder.parameters(),
    lr = dec_lr,
    eps = ADAM_EPS
)
enc_opt = torch.optim.Adam(
    encoder.parameters(),
    lr = enc_lr
)
dec_sch = torch.optim.lr_scheduler.StepLR(
    dec_opt,
    step_size = dec_step,
    gamma = dec_decay
)
# lr_lbmd = lambda _: max(
#     0.7
#     ** (
#         int(
#             self.global_step
#             * batch_size
#             / 2e4
#         )
#     ),
#     1e-5,
# )
# enc_sch = torch.optim.lr_scheduler.LambdaLR(enc_opt, lr_lambda=lr_lbmd)
enc_sch = torch.optim.lr_scheduler.StepLR(
    enc_opt,
    step_size = enc_step,
    gamma = enc_decay
)
loss_config = getLossConfig()

print('training ...')

for epoch in range(100000):
    decoder.train()
    encoder.train()
    ep_result = {}
    bc = 0.
    for batch in train_dataset:
        bc += 1.
        shape, points, ind = batch[0]
        shape_result = run_train_decoder(
            shape, points, encoder, decoder
        )
        loss = 0.
        if len(shape_result) > 0:
            for key in loss_config:
                shape_result[key] *= loss_config[key]
                if torch.is_tensor(shape_result[key]):
                    loss += shape_result[key]

            for key in shape_result:
                res = shape_result[key]
                if torch.is_tensor(res):
                    res = res.detach()
                if key not in ep_result:
                    ep_result[key] = res
                else:
                    ep_result[key] += res

        if torch.is_tensor(loss) and enc_opt is not None and dec_opt is not None:
            dec_opt.zero_grad()
            enc_opt.zero_grad()
            loss.backward()
            dec_opt.step()
            enc_opt.step()

    if (epoch + 1) % 10 == 0:
        print_train_results(ep_result)
        with torch.no_grad():
            eval_results = model_eval(eval_val_dataset, encoder, decoder, "train_out", "0", epoch)
            print_eval_results(eval_results, "val")
            eval_results = model_eval(eval_train_dataset, encoder, decoder, "train_out", "0", epoch)
            print_eval_results(eval_results, "train")
        torch.save(encoder.state_dict(), "train_out/encoder.pt")
        torch.save(decoder.state_dict(), "train_out/decoder.pt")


    dec_sch.step()
    enc_sch.step()
