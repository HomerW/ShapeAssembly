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
from inference_model import model_eval, get_partnet_data, get_random_data, prog_to_pc
from print_fun import print_train_results, print_eval_results, getLossConfig

num_samps = 10000
hidden_dim = 256
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
MAX_PDEPTH = 10
MAX_CUBOIDS = 10

def round_prog(prog):
    P = Program()
    new_lines = []
    for l in prog['prog']:
        if "Cuboid" in l:
            parse = P.parseCuboid(l)
            new_num = [round(x.item(), 3) for x in parse[1:4]]
            new_lines.append(f"{parse[0]} = Cuboid({new_num[0]}, {new_num[1]}, {new_num[2]}, {parse[4]})")
        elif "attach" in l:
            parse = P.parseAttach(l)
            new_num = [round(x.item(), 3) for x in parse[2:]]
            new_lines.append(f"attach({parse[0]}, {parse[1]}, {new_num[0]}," +
                                      f" {new_num[1]}, {new_num[2]}, {new_num[3]}, {new_num[4]}, {new_num[5]})")
        elif "squeeze" in l:
            parse = P.parseSqueeze(l)
            new_num = [round(x.item(), 3) for x in parse[-2:]]
            new_lines.append(f"squeeze({parse[0]}, {parse[1]}, {parse[2]}," +
                                      f" {parse[3]}, {new_num[0]}, {new_num[1]})")
        elif "translate" in l:
            parse = P.parseTranslate(l)
            new_num = [round(x.item(), 3) for x in parse[-1:]]
            new_lines.append(f"translate({parse[0]}, {parse[1]}, {parse[2]}, {new_num[0]})\n")
        else:
            new_lines.append(l)
    prog['prog'] = new_lines
    for c in prog["children"]:
        if not c == {}:
            round_prog(c)

def check_size(prog, level):
    if level > MAX_PDEPTH:
        return False
    if len(prog["children"]) > MAX_CUBOIDS:
        return False

    return all([check_size(c, level + 1) for c in prog["children"] if not c == {}])



def train_inference(encoder, enc_opt, decoder, dec_opt, dataset, max_epochs=None):
    if max_epochs is None:
        epochs = 100
    else:
        epochs = max_epochs
    train_dataset, eval_train_dataset, eval_val_dataset = dataset

    best_encoder_dict = encoder.state_dict()
    best_decoder_dict = decoder.state_dict()
    best_fscore = -1
    patience = 3

    loss_config = getLossConfig()

    for epoch in range(epochs):
        decoder.train()
        encoder.train()
        ep_result = {}
        bc = 0.
        for batch in train_dataset:
            bc += 1.
            shape, points, ind = batch[0]
            points = points.to(device).unsqueeze(0)
            encoding = encoder(points).unsqueeze(0)
            shape_result = run_train_decoder(
                shape, encoding, decoder
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
            print_train_results(ep_result, bc)
            with torch.no_grad():
                eval_results, _ = model_eval(eval_val_dataset, encoder, decoder, "ws_out", "val", epoch)
                val_fscore = eval_results['fscores']
                print_eval_results(eval_results, "M-phase val")
                # eval_results = model_eval(eval_train_dataset, encoder, decoder, "train_out", "train", epoch)
                # print_eval_results(eval_results, "train")

            if val_fscore < best_fscore:
                num_worse += 1
            else:
                num_worse = 0
                best_fscore = val_fscore
                best_encoder_dict = encoder.state_dict()
                best_decoder_dict = decoder.state_dict()
            if num_worse >= patience:
                # load the best model and stop training
                encoder.load_state_dict(best_encoder_dict)
                decoder.load_state_dict(best_decoder_dict)
                break


def infer_programs(encoder, decoder):
    train_dataset, val_dataset, eval_train_dataset, eval_val_dataset = get_partnet_data("data/chair", "chair", 5000)
    with torch.no_grad():
        eval_results, train_progs = model_eval(train_dataset, encoder, decoder, "ws_out", "val", 0)
        print_eval_results(eval_results, "E-phase train")
        eval_results, eval_val_progs = model_eval(eval_val_dataset, encoder, decoder, "ws_out", "train", 0)
        print_eval_results(eval_results, "E-phase val")

    train_samples = []
    eval_val_samples = []
    for ind, prog in train_progs:
        if not check_size(prog, 1):
            continue
        try:
            round_prog(prog)
            prog_pc = prog_to_pc(prog)
            nprog = progToData(prog)
            train_samples.append((nprog, prog_pc, ind))
        except:
            pass


    for ind, prog in eval_val_progs:
        if not check_size(prog, 1):
            continue
        try:
            round_prog(prog)
            prog_pc = prog_to_pc(prog)
            nprog = progToData(prog)
            eval_val_samples.append((nprog, prog_pc, ind))
        except:
            pass

    print(f"LEN TRAIN: {len(train_samples)}")
    print(f"LEN VAL: {len(eval_val_samples)}")

    train_dataset = DataLoader(train_samples, batch_size, shuffle=True, collate_fn = _col)
    eval_train_dataset = DataLoader(train_samples[:num_eval], batch_size=1, shuffle=False, collate_fn = _col)
    eval_val_dataset = DataLoader(eval_val_samples, batch_size=1, shuffle = False, collate_fn = _col)

    return (train_dataset, eval_train_dataset, eval_val_dataset)

def wake_sleep(iterations):
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

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

    encoder.load_state_dict(torch.load("train_out/encoder-256-random.pt"))
    decoder.load_state_dict(torch.load("train_out/decoder-256-random.pt"))

    for iter in range(iterations):
        dataset = infer_programs(encoder, decoder)
        train_inference(encoder, enc_opt, decoder, dec_opt, dataset)

        torch.save(encoder.state_dict(), "train_out/encoder-256-ws.pt")
        torch.save(decoder.state_dict(), "train_out/decoder-256-ws.pt")


wake_sleep(1000)
