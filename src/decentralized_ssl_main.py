#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import random
import csv
import numpy as np
from tqdm import tqdm
import torch

from tensorboardX import SummaryWriter
from options import args_parser
from models import *
from utils import *
from datetime import datetime
from update import LocalUpdate, GlobalUpdate, test_inference
from pprint import pprint
import IPython

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import RandomSampler
import socket

def comm_quan(w, bitw):
    if bitw < 32:
        return dorefa_quan(w, bitw)
    else:
        return w

def dorefa_quan(w, bitw):
    scale = 2 ** bitw - 1
    for key in w.keys():
        if torch.max(torch.abs(w[key])) == 0:
            continue
        elif "weight" not in key:
            continue
        w_key = torch.tanh(w[key])
        w_key = w_key / (2*torch.max(torch.abs(w_key))) + 0.5
        w_key = 2*torch.round(w_key*scale)/scale - 1
        w[key] = w_key
    return w

def unif_quan(w, bitw):
    scale = 2 ** bitw - 1
    for key in w.keys():
        if torch.max(torch.abs(w[key])) == 0:
            continue
        elif "weight" not in key:
            continue
        w_key = w[key]
        w_key = w_key / (2*torch.max(torch.abs(w_key))) + 0.5
        w_key = 2*torch.round(w_key*scale)/scale - 1
        w[key] = w_key
    return w

if __name__ == "__main__":
    start_time = time.time()

    # define paths
    path_project = os.path.abspath("..")
    args = args_parser()
    exp_details(args)

    if args.distributed_training:
        global_rank, world_size = get_dist_env()
        hostname = socket.gethostname()
        print("initing distributed training")
        dist.init_process_group(
            backend="nccl",
            rank=global_rank,
            world_size=world_size,
            init_method=args.dist_url,
        )
        args.world_size = world_size
        args.batch_size *= world_size

    device = "cuda" if args.gpu else "cpu"

    # load dataset and user groups
    set_seed(args.seed)
    (
        train_dataset,
        test_dataset,
        public_train_dataset,
        user_groups,
        memory_dataset,
        test_user_groups,
    ) = get_dataset(args)
    batch_size = args.batch_size
    pprint(args)

    model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S") + "_{}".format(
        str(os.getpid())
    )
    model_output_dir = "save/" + model_time
    # model_output_dir = "save/08_05_2023_12:21:04_20043"
    args.model_time = model_time
    save_args_json(model_output_dir, args)
    logger = SummaryWriter(model_output_dir + "/tensorboard")
    args.start_time = datetime.now()

    # build model
    start_epoch = 0
    cbit = 32
    global_model = get_global_model(args, cbit, train_dataset).to(device)
    # load_global_weights = torch.load(model_output_dir + "/model.pth")
    # global_model.load_state_dict(load_global_weights["model"])
    global_weights = global_model.state_dict()

    if args.distributed_training:
        global_model = DDP(global_model)
    else:
        global_model = torch.nn.DataParallel(global_model)
    global_model.train()

    # Training
    train_loss, train_accuracy, global_model_accuracy = [], [], []
    print_every = 200
    # local_models = [copy.deepcopy(global_model) for _ in range(args.num_users)]
    local_models = []
    # cbit_local = [3, 4, 4, 5, 6]
    # cbit_local = [4, 4, 4, 4, 4 ]
    cbit_local = [32, 32, 32, 32, 32]
    for idx in range(args.num_users):
        local_model = get_local_model(args, cbit_local[idx]).to(device)

        if args.distributed_training:
            local_model = DDP(local_model)
        else:
            local_model = torch.nn.DataParallel(local_model)
        local_model.train()
        local_weights = local_model.state_dict()
        # print("local model params:", local_weights.keys())
        # print("global_weights: ", global_weights.keys())
        for key1, key2 in zip(local_weights.keys(), global_weights.keys()):
            local_weights[key1].data = global_weights[key2].data
        local_models.append(local_model)


    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            global_model.parameters(), lr=args.lr, weight_decay=1e-6
        )
    else:
        train_lr = (
            args.lr * args.world_size * (args.batch_size / 256)
            if args.distributed_training
            else args.lr
        )
        optimizer = torch.optim.SGD(global_model.parameters(), lr=train_lr)

    total_epochs = int(args.epochs / args.local_ep)  # number of rounds
    schedule = [
        int(total_epochs * 0.3),
        int(total_epochs * 0.6),
        int(total_epochs * 0.9),
    ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=schedule, gamma=0.3
    )
    print("output model:", model_output_dir)
    print(
        "number of users per round: {}".format(max(int(args.frac * args.num_users), 1))
    )
    print("total number of rounds: {}".format(total_epochs))
    local_update_clients = [
        LocalUpdate(
            args=args,
            dataset=train_dataset,
            idx=idx,
            idxs=user_groups[idx],
            logger=logger,
            output_dir=model_output_dir,
        )
        for idx in range(args.num_users)
    ]

    idx = 0
    for client in local_update_clients:
        client.init_model(local_models[idx])
        idx += 1

    global_quan_update = GlobalUpdate(
            args=args,
            dataset=public_train_dataset,
            logger=logger,
            output_dir=model_output_dir,
    )
    global_quan_update.init_model(global_model)


    lr = optimizer.param_groups[0]["lr"]

    print("start epoch:", start_epoch, total_epochs)
    for epoch in tqdm(range(start_epoch, total_epochs)):
        # print("start epoch:", start_epoch, epoch)

        local_weights, local_losses = [], []
        print(f"\n | Global Training Round : {epoch+1} | Model : {model_time}\n")

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # update each local model
        for idx in idxs_users:
            local_model = local_update_clients[idx]
            w, loss = local_model.update_ssl_weights(
                model=local_models[idx],
                global_round=epoch,
                gbit = int(cbit_local[idx]+2),
                cbit = cbit_local[idx],
                lr=lr,
            )
            local_models[idx] = local_model.model
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        
        # Collect trained local weights in the last epoch
        if epoch == total_epochs - 1:
            print("Specify local models")
            local_models_n = [copy.deepcopy(local_models[i]) for i in range(args.num_users)]

        local_weights = [dorefa_quan(local_weights[idx], cbit_local[idx]) for idx in range(args.num_users)]
        
        # De-quantization


        # update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        global_weights = dorefa_quan(global_weights, cbit)

        # Re-quantization
        # global_model_q = global_quan_update
        # w, loss = global_model_q.update_ssl_weights(
        #     model = copy.deepcopy(global_model),
        #     global_round = epoch,
        #     lr = lr,
        #     cbit = cbit_local[idx]
        # )
        # global_model = global_model_q.model
        # global_weights = global_model_q.model.state_dict()


        if args.average_without_bn:
            for i in range(args.num_users):
                local_models[i] = load_weights_without_batchnorm(
                    local_models[i], global_weights
                )
        else:
            for i in range(args.num_users):
                local_models[i] = load_weights(local_models[i], global_weights)
        global_model.load_state_dict(global_weights)

        # print global training loss after every 'i' rounds
        if (int(epoch * args.local_ep) + 1) % print_every == 0:
            print(f" \nAvg Training Stats after {epoch+1} global rounds:")
            print(f"Training Local Client Loss : {np.mean(np.array(train_loss))}")

        scheduler.step()
        lr = scheduler._last_lr[0]
        global_model.module.save_model(model_output_dir, step=epoch)

    global_model.module.save_model(model_output_dir, step=epoch)

    # evaluate local representations
    args.finetuning_epoch = 100
    for idx in range(args.num_users):
        train_dataset_idx, _, test_dataset_idx, _ = local_update_clients[idx].get_dataset()
        # test_acc_l = global_repr_local_classifier(args, global_model, train_dataset_idx, test_dataset_idx, args.finetuning_epoch)
        test_acc_l = global_repr_local_classifier(args, cbit_local[idx], local_models_n[idx], train_dataset_idx, test_dataset_idx, args.finetuning_epoch)
        print("client", idx, cbit_local[idx], test_acc_l)

    # evaluate representations
    print("evaluating representations: ", model_output_dir)
    test_acc = global_repr_global_classifier(args, cbit, global_model, args.finetuning_epoch)


    print(f" \n Results after {args.epochs} global rounds of training:")
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))

    # PLOTTING (optional)
    pprint(args)
    suffix = "{}_{}_{}_{}_dec_ssl_{}".format(
        args.model, args.batch_size, args.epochs, args.save_name_suffix, args.ssl_method
    )
    write_log_and_plot(model_time, model_output_dir, args, suffix, test_acc)
