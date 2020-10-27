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
from pointnet_fixed import PointNetEncoder as PCEncoder
# from pc_encoder import PCEncoder
from model_prog import FDGRU, load_progs, progToData, getInds, _col, \
                                run_train_decoder, run_eval_decoder, run_eval_decoder_beam, run_eval_decoder_beam2
from ShapeAssembly import hier_execute, Program
from print_fun import print_train_results, print_eval_results

num_samps = 10000
hidden_dim = 1024
batch_size = 1
num_eval = 100
ADAM_EPS = 1e-6
dec_lr = 0.0001
enc_lr = 0.0001
dec_step = 5000
dec_decay = 1.0
enc_step = 5000
enc_decay = 1.0
device = torch.device("cuda")

def prog_to_pc(prog):
    verts, faces = hier_execute(prog)
    for i in range(3):
        verts[:,i] = verts[:,i] - verts[:,i].mean()
    pc = utils.sample_surface(faces, verts.unsqueeze(0), num_samps, return_normals=False)[0]
    return pc

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

def get_partnet_data(dataset_path, category, max_shapes):
    raw_indices, progs = load_progs(dataset_path, max_shapes)

    inds_and_progs = list(zip(raw_indices, progs))
    random.shuffle(inds_and_progs)

    inds_and_progs = inds_and_progs[:max_shapes]

    print('Converting progs to tensors')

    samples = []
    prog_lens = 0
    num_children = 0
    for ind, prog in inds_and_progs:
        prog_lens += len(prog['prog'])
        num_children += len(prog['children'])
        # prog_pc = lines_to_pc(prog['prog'])
        prog_pc = prog_to_pc(prog)
        # prog['children'] = [{}] * len(prog['children'])
        nprog = progToData(prog)
        samples.append((nprog, prog_pc, ind))
    print(f"AVERAGE PROG LENGTH: {prog_lens/len(inds_and_progs)}")
    print(f"AVERAGE NUM CHILDREN: {num_children/len(inds_and_progs)}")

    train_ind_file = f'data_splits/{category}/train.txt'
    val_ind_file = f'data_splits/{category}/val.txt'

    train_samples = []
    val_samples = []

    train_inds = getInds(train_ind_file)
    val_inds = getInds(val_ind_file)

    misses = 0.

    for (prog, prog_pc, ind) in samples:
        if ind in train_inds:
            train_samples.append((prog, prog_pc, ind))
        elif ind in val_inds:
            val_samples.append((prog, prog_pc, ind))
        else:
            assert False, "idk what this is about"
            # if keep_missing:
            #     kept += 1
            #     if random.random() < holdout_perc:
            #         val_samples.append((prog, ind))
            #     else:
            #         train_samples.append((prog, ind))
            # else:
            #     misses += 1

    train_num = len(train_samples)
    val_num = len(val_samples)
    print(f"Training size: {train_num}")
    print(f"Validation size: {val_num}")

    train_dataset = DataLoader(train_samples, batch_size, shuffle=True, collate_fn = _col)
    eval_train_dataset = DataLoader(train_samples[:num_eval], batch_size=1, shuffle=False, collate_fn = _col)
    val_dataset = DataLoader(val_samples, batch_size, shuffle = False, collate_fn = _col)
    eval_val_dataset = DataLoader(val_samples[:num_eval], batch_size=1, shuffle = False, collate_fn = _col)

    return train_dataset, val_dataset, eval_train_dataset, eval_val_dataset

def get_random_data(dataset_path, max_shapes, train_num):
    raw_indices, progs = load_progs(dataset_path, max_shapes)

    inds_and_progs = list(zip(raw_indices, progs))
    random.shuffle(inds_and_progs)

    inds_and_progs = inds_and_progs[:max_shapes]

    print('Converting progs to tensors')

    samples = []
    prog_lens = 0
    num_children = 0
    for ind, prog in inds_and_progs:
        prog_lens += len(prog['prog'])
        num_children += len(prog['children'])
        # prog_pc = lines_to_pc(prog['prog'])
        prog_pc = prog_to_pc(prog)
        # prog['children'] = [{}] * len(prog['children'])
        nprog = progToData(prog)
        samples.append((nprog, prog_pc, ind))
    print(f"AVERAGE PROG LENGTH: {prog_lens/len(inds_and_progs)}")
    print(f"AVERAGE NUM CHILDREN: {num_children/len(inds_and_progs)}")

    train_samples = samples[:train_num]
    val_samples = samples[train_num:]

    train_num = len(train_samples)
    val_num = len(val_samples)
    print(f"Training size: {train_num}")
    print(f"Validation size: {val_num}")

    train_dataset = DataLoader(train_samples, batch_size, shuffle=True, collate_fn = _col)
    eval_train_dataset = DataLoader(train_samples[:num_eval], batch_size=1, shuffle=False, collate_fn = _col)
    val_dataset = DataLoader(val_samples, batch_size, shuffle = False, collate_fn = _col)
    eval_val_dataset = DataLoader(val_samples[:num_eval], batch_size=1, shuffle = False, collate_fn = _col)

    return train_dataset, val_dataset, eval_train_dataset, eval_val_dataset

# Runs an epoch of evaluation logic
def model_eval(dataset, encoder, decoder, outpath, exp_name, epoch, beam_width = None):
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

        points = points.to(device).unsqueeze(0)
        encoding = encoder(points).unsqueeze(0)

        if beam_width is None:
            prog, shape_result = run_eval_decoder(
                encoding, decoder, False, shape
            )
        else:
            results = run_eval_decoder_beam(
                encoding, decoder, False, shape, beam_width
            )
            best_per = -1
            best_prog = None
            best_shape_result = None
            for prog, shape_result in results:
                # recon_results, recon_misses = metrics.recon_metrics(
                #     [(prog, shape, ind)], outpath, exp_name, "generated", epoch, True
                # )
                if shape_result['cmdc'] > best_per:
                    best_prog = prog
                    best_shape_result = shape_result
                    best_per = shape_result['cmdc']
            prog = best_prog
            shape_result = best_shape_result


        # print(named_results[f'count'])
        # def print_recur(prog):
        #     lines = 0
        #     for p in prog['prog']:
        #         print(p)
        #         lines += 1
        #     for c in prog['children']:
        #         if not c == {}:
        #             lines += print_recur(c)
        #     return lines
        # tot_lines = print_recur(prog)
        # print(f"NUM LINES: {tot_lines}")
        # print(f"IND: {ind}")

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
        recon_sets, outpath, exp_name, "generated", epoch, True
    )

    for key in recon_results:
        named_results[key] = recon_results[key]

    named_results[f'miss_hier_prog'] += recon_misses

    named_results[f'prog_creation_perc'] = (
        named_results[f'count'] - named_results[f'miss_hier_prog']
    ) / named_results[f'count']

    return named_results

RANDOM_SEED = 39
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

train_dataset, val_dataset, eval_train_dataset, eval_val_dataset = get_partnet_data("data/chair", "chair", 10000)
# train_dataset, val_dataset, eval_train_dataset, eval_val_dataset = get_random_data("random_data", 1000, 900)

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
    
enc_sch = torch.optim.lr_scheduler.StepLR(
    enc_opt,
    step_size = enc_step,
    gamma = enc_decay
)

print('training ...')

with torch.no_grad():
    encoder.load_state_dict(torch.load("train_out/encoder-1024-hier.pt"))
    decoder.load_state_dict(torch.load("train_out/decoder-1024-hier.pt"))
    eval_results = model_eval(eval_val_dataset, encoder, decoder, "train_out", "val", 0, beam_width=5)
    print_eval_results(eval_results, "val")

# for epoch in range(100000):
#     decoder.train()
#     encoder.train()
#     ep_result = {}
#     bc = 0.
#     for batch in train_dataset:
#         bc += 1.
#         shape, points, ind = batch[0]
#         points = points.to(device).unsqueeze(0)
#         encoding = encoder(points).unsqueeze(0)
#         shape_result = run_train_decoder(
#             shape, encoding, decoder
#         )
#         loss = 0.
#         if len(shape_result) > 0:
#             for key in loss_config:
#                 shape_result[key] *= loss_config[key]
#                 if torch.is_tensor(shape_result[key]):
#                     loss += shape_result[key]
#
#             for key in shape_result:
#                 res = shape_result[key]
#                 if torch.is_tensor(res):
#                     res = res.detach()
#                 if key not in ep_result:
#                     ep_result[key] = res
#                 else:
#                     ep_result[key] += res
#
#         if torch.is_tensor(loss) and enc_opt is not None and dec_opt is not None:
#             dec_opt.zero_grad()
#             enc_opt.zero_grad()
#             loss.backward()
#             dec_opt.step()
#             enc_opt.step()
#
#     if (epoch + 1) % 10 == 0:
#         print_train_results(ep_result)
#         with torch.no_grad():
#             eval_results = model_eval(eval_val_dataset, encoder, decoder, "train_out", "val", epoch)
#             print_eval_results(eval_results, "val")
#             eval_results = model_eval(eval_train_dataset, encoder, decoder, "train_out", "train", epoch)
#             print_eval_results(eval_results, "train")
#         torch.save(encoder.state_dict(), "train_out/encoder-256-flat-just-attach-random.pt")
#         torch.save(decoder.state_dict(), "train_out/decoder-256-flat-just-attach-random.pt")
#
#
#     dec_sch.step()
#     enc_sch.step()
