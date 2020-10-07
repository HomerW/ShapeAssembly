import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import losses
from parse_prog import progToTarget, predToProg
from argparse import Namespace
from ShapeAssembly import hier_execute
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import argparse
from tqdm import tqdm
import ast
import metrics
from sem_valid_reencode import semValidGen
from copy import deepcopy

"""
Modeling logic for a generative model of ShapeAssembly Programs.

Encoder is defined in ENCGRU. Decoder is defined in FDGRU.

run_train_decoder has the core logic for training in a teacher-forced manner
run_eval_decoder has the core logic for evaluating in an auto-regressive manner

run_train is the "main" entrypoint.

"""

outpath = "model_output"
device = torch.device("cuda")
#device = torch.device("cpu")

INPUT_DIM = 63 # tensor representation of program. Check ProgLoss in utils for a detailed comment of how lines in ShapeAssembly are represented as Tensors
MAX_PLINES = 100 # Cutoff number of lines in eval hierarchical program
MAX_PDEPTH = 10 # Cutoff number of programs in eval hierarchical program
ADAM_EPS = 1e-6 # increase numerical stability
VERBOSE = True
SAVE_MODELS = True
num_samps = 5000

# Program reconstruction loss logic
fploss = losses.ProgLoss()
closs = torch.nn.BCEWithLogitsLoss()

# A 'tokenization' of the line command
def cleanCommand(struct):
    assert struct.shape[1] == 1, 'bad input to clean command'
    struct = struct.squeeze()
    new = torch.zeros(7, dtype=torch.float).to(device)

    c = torch.argmax(struct[:7])
    new[c] = 1.0

    new = new.unsqueeze(0).unsqueeze(0)

    return new

# A 'tokenization' of the line cube indices
def cleanCube(struct):
    assert struct.shape[1] == 1, 'bad input to clean cube'
    struct = struct.squeeze()
    new = torch.zeros(33, dtype=torch.float).to(device)

    c1 = torch.argmax(struct[:11])
    new[c1] = 1.0

    c2 = torch.argmax(struct[11:22])
    new[11+c2] = 1.0

    c3 = torch.argmax(struct[22:33])
    new[22+c3] = 1.0

    new = new.unsqueeze(0).unsqueeze(0)

    return new


def getLossConfig():
    loss_config = {
        'cmd': 1.,
        'cub_prm': 50.,

        'xyz_prm': 50.,
        'uv_prm': 50.,
        'sym_prm': 50.,

        'cub': 1.,
        'sym_cub': 1.,
        'sq_cub': 1.,

        'bb': 50.,

        'axis': 1.,
        'face': 1.,
        'align': 1.
    }

    return loss_config

# Multi-layer perceptron helper function
class MLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x),.2)
        x = F.leaky_relu(self.l2(x),.2)
        return self.l3(x)


# GRU recurrent Decoder
class FDGRU(nn.Module):
    def __init__(self, hidden_dim):
        super(FDGRU, self).__init__()

        self.bbdimNet = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first = True)
        self.init_gru = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.inp_net = MLP(
            INPUT_DIM + hidden_dim + 3,
            hidden_dim,
            hidden_dim,
            hidden_dim
        )

        self.cmd_net = MLP(hidden_dim, hidden_dim // 2, hidden_dim //4, 7)

        self.cube_net = MLP(hidden_dim + 7, hidden_dim //2, hidden_dim //4, 33)

        self.dim_net = MLP(
            hidden_dim + 3,
            hidden_dim // 2,
            hidden_dim // 4,
            3
        )

        self.align_net = MLP(
            hidden_dim + 3,
            hidden_dim // 2,
            hidden_dim // 4,
            1
        )

        self.att_net = MLP(
            hidden_dim + 22,
            hidden_dim,
            hidden_dim,
            6
        )

        # take in cuboids involved -> predict the face and uv
        self.squeeze_net = MLP(
            hidden_dim + 33,
            hidden_dim // 2,
            hidden_dim // 4,
            8
        )

        self.sym_net = MLP(
            hidden_dim + 3 + 11,
            hidden_dim // 2,
            hidden_dim // 4,
            5
        )

    # Inp squence is the input sequence, h is the current hidden state, h_start is the hidden state at the beginning of the local program
    # bb_dims are the dimensions of the bbox, hier_ind is the depth of the hierarchy. gt_struct_seq contains the target cubes + command information for further-teacher forcing during training
    def forward(self, inp_seq, reencoding, h, bb_dims, gt_struct_seq=None):

        bb_dims = bb_dims.unsqueeze(0).unsqueeze(0).repeat(1,inp_seq.shape[1],1)

        inp = self.inp_net(
            torch.cat(
                (inp_seq, reencoding, bb_dims), dim=2)
        )

        gru_out, h = self.gru(inp, h)

        cmd_out = self.cmd_net(gru_out)

        if gt_struct_seq is not None:
            clean_cmd = gt_struct_seq[:,:,:7]
        else:
            clean_cmd = cleanCommand(cmd_out)

        cube_out = self.cube_net(
            torch.cat((gru_out, clean_cmd), dim = 2)
        )

        if gt_struct_seq is not None:
            clean_cube = gt_struct_seq[:,:,7:40]
        else:
            clean_cube = cleanCube(cube_out)

        dim_out = self.dim_net(
            torch.cat((gru_out, bb_dims), dim = 2)
        )

        align_out = self.align_net(
            torch.cat((gru_out, bb_dims), dim = 2)
        )

        att_out = self.att_net(
            torch.cat((gru_out, clean_cube[:,:,:22]), dim = 2)
        )

        sym_out = self.sym_net(
            torch.cat((gru_out, clean_cube[:,:,:11], bb_dims), dim =2)
        )

        squeeze_out = self.squeeze_net(
            torch.cat((gru_out, clean_cube), dim=2)
        )

        out = torch.cat(
            (cmd_out, cube_out, dim_out, att_out, sym_out, squeeze_out, align_out), dim=2
        )

        return out, h

# Logic for training the decoder (decoder) on a single hierarchical program prog (we don't use batching because hierarchical data +
# batching = hard). h0 is the latent code to be decoded
def run_train_decoder(prog, pc, encoder, decoder):

    shape_result = {}

    reencoding = encoder(pc).unsqueeze(0)

    bb_pred = decoder.bbdimNet(reencoding[:, 0])
    bb_loss = (bb_pred - prog["bb_dims"]).abs().sum()

    shape_result['bb'] = bb_loss

    num_lines = 0.

    children = prog["children"]
    inp_seq = prog["inp"].transpose(0,1)
    tar_seq = prog["tar"].transpose(0,1)
    weights = prog["weights"].unsqueeze(0)
    bb_dims = prog["bb_dims"]

    num_lines += inp_seq.shape[1]

    # Teacher forcing the decoder
    pout, _ = decoder(
        inp_seq, reencoding, decoder.init_gru, bb_dims, tar_seq[:,:,:40]
    )

    # This is the core reconstruction loss calculation
    prog_result = fploss(
        pout,
        tar_seq,
        weights
    )

    for key in prog_result:
        if key not in shape_result:
            shape_result[key] = prog_result[key]
        else:
            shape_result[key] += prog_result[key]

    cub_inds = (
        torch.argmax(tar_seq[:, :, :7], dim = 2) == 1
    ).squeeze().nonzero().squeeze()

    shape_result['nl'] = num_lines

    return shape_result

# Decode latent code in a hierarchical shapeAssembly program using decoder in an auto-regressive manner.
def run_eval_decoder(pc, encoder, decoder, rejection_sample, gt_prog = None):

    index = 0

    prog = {
        "children": [],
        "bb_dims": None,
        "prog": []
    }

    num_lines = 0.
    all_preds = []

    shape_result = {
        'corr_line_num': 0.,
        'bad_leaf': 0.
    }

    prog["name"] = index
    index += 1

    if gt_prog is None or len(gt_prog) == 0:
        shape_result['bad_leaf'] += 1.

    # Semantic validity logic that handles local program creation
    preds, prog_out = semValidGen(
        pc, prog, encoder, decoder, MAX_PLINES, INPUT_DIM, device, gt_prog, rejection_sample
    )

    num_lines += len(preds)

    # Logic for calculating loss / metric performance in eval mode
    if gt_prog is not None and len(gt_prog) > 0:
        bb_loss = (prog["bb_dims"] - gt_prog["bb_dims"]).abs().sum()
        shape_result['bb'] = bb_loss

        gt_tar_seq = gt_prog["tar"].transpose(0,1)
        gt_weights = gt_prog["weights"].unsqueeze(0)
        gt_bb_dims = gt_prog["bb_dims"]

        try:
            if len(prog_out) > 1:
                prog_out = torch.cat([p for p in prog_out], dim = 1)
            else:
                prog_out = torch.zeros(1,1,INPUT_DIM).to(device)

            if prog_out.shape[1] == gt_tar_seq.shape[1]:
                shape_result['corr_line_num'] += 1.

            prog_result = fploss(
                prog_out[:,:gt_tar_seq.shape[1],:],
                gt_tar_seq,
                gt_weights
            )

            for key in prog_result:
                if key not in shape_result:
                    shape_result[key] = prog_result[key]
                else:
                    shape_result[key] += prog_result[key]

        except Exception as e:
            if VERBOSE:
                print(e)
            pass

    all_preds.append(preds)

    shape_result['nl'] = num_lines

    try:
        getHierProg(prog, all_preds)
        return prog, shape_result

    except Exception as e:
        if VERBOSE:
            print(f"FAILED IN EVAL DECODER WITH {e}")
        return None, shape_result

# Given the decoder's predictions, create a well-structured ShapeAssembly Program (in text)
def getHierProg(hier_prog, all_preds):
    if len(hier_prog) == 0:
        return
    prog = predToProg(all_preds[hier_prog["name"]])
    hier_prog["prog"] = prog
    for c in hier_prog["children"]:
        getHierProg(c, all_preds)

# convert a text based hierarchical program into a tensorized version
def progToData(prog):
    if len(prog) == 0:
        return {}

    inp, tar, weights, bb_dims = progToTarget(prog["prog"])
    prog["inp"] = inp.unsqueeze(1).cpu()
    prog["tar"] = tar.unsqueeze(1).cpu()
    prog["weights"] = weights.cpu()
    prog["children"] = [progToData(c) for c in prog["children"]]
    prog["bb_dims"] = bb_dims.cpu()

    return prog

# Dummy collate function
def _col(samples):
    return samples

# Used to re-start training from a previous run
def loadConfigFile(exp_name):
    args = None
    with open(f"{outpath}/{exp_name}/config.txt") as f:
        for line in f:
            args = eval(line)

    assert args is not None, 'failed to load config'
    return args

# Set-up new experiment directory
def writeConfigFile(args):
    os.system(f'mkdir {outpath} > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name} > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/plots > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/plots/train > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/plots/eval > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/plots/gen > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs/train > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs/val > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs/gen > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/programs/gt > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs/train > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs/val > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs/gen > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/objs/gt > /dev/null 2>&1')
    os.system(f'mkdir {outpath}/{args.exp_name}/models > /dev/null 2>&1')
    with open(f"{outpath}/{args.exp_name}/config.txt", "w") as f:
        f.write(f"{args}\n")

# Load a dataset of hierarchical ShapeAssembly Programs
def load_progs(dataset_path, max_shapes):
    inds = os.listdir(dataset_path)
    inds = [i.split('.')[0] for i in inds[:max_shapes]]
    good_inds = []
    progs = []
    for ind in tqdm(inds):
        hp = utils.loadHPFromFile(f'{dataset_path}/{ind}.txt')
        if hp is not None and len(hp) > 0:
            progs.append(hp)
            good_inds.append(ind)
    return good_inds, progs

# Helper function for keeping consistent train/val splits
def getInds(train_ind_file):
    inds = set()
    with open(train_ind_file) as f:
        for line in f:
            inds.add(line.strip())
    return inds
