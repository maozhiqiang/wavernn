#!/usr/bin/env python
# -*- coding:utf-8 -*-
from model import WaveRNN
import torch
from torch.autograd import Variable
import numpy as np
import time
from torch import optim
import torch.nn.functional as F
import sys
from hparam import Hyperparams as hp
from utils import split_signal
import math


def sine_wave(freq, length, sample_rate=hp.sr):
    return np.sin(np.arange(length) * 2 * math.pi * freq / sample_rate).astype(np.float32)


model = WaveRNN().cuda()
x = sine_wave(freq=500, length=hp.sr * 30)


def input_split():
    coarse_classes, fine_classes = split_signal(x)
    coarse_classes = np.reshape(coarse_classes, (1, -1))
    fine_classes = np.reshape(fine_classes, (1, -1))
    return coarse_classes, fine_classes


coarse_classes, fine_classes = input_split()


def train(model, optimizer, coarse_classes, fine_classes,num_steps, seq_len=960):
    start = time.time()
    running_loss = 0

    for step in range(num_steps):
        optimizer.zero_grad()
        loss = wavernn_loss(model, coarse_classes, fine_classes, seq_len)

        running_loss += (loss.data[0] / seq_len)
        loss.backward()
        optimizer.step()

        speed = (step + 1) / (time.time() - start)

        sys.stdout.write('\rStep: %i/%i --- NLL: %.2f --- Speed: %.3f batches/second ' %
                         (step + 1, num_steps, running_loss / (step + 1), speed))


def wavernn_loss(model, coarse_classes, fine_classes, seq_len):
    loss = 0
    hidden = model.init_hidden()
    rand_idx = np.random.randint(0, coarse_classes.shape[1] - seq_len - 1)
    for i in range(seq_len):
        j = rand_idx + i

        x_coarse = coarse_classes[:, j:j + 1]
        x_fine = fine_classes[:, j:j + 1]
        x_input = np.concatenate([x_coarse, x_fine], axis=1)
        x_input = x_input / 127.5 - 1.
        x_input = Variable(torch.FloatTensor(x_input)).cuda()

        y_coarse = coarse_classes[:, j + 1]
        y_fine = fine_classes[:, j + 1]
        y_coarse = Variable(torch.LongTensor(y_coarse)).cuda()
        y_fine = Variable(torch.LongTensor(y_fine)).cuda()

        current_coarse = y_coarse.float() / 127.5 - 1.
        current_coarse = current_coarse.unsqueeze(-1)

        out_coarse, out_fine, hidden = model(x_input, hidden, current_coarse)

        loss_coarse = F.cross_entropy(out_coarse, y_coarse)
        loss_fine = F.cross_entropy(out_fine, y_fine)
        loss += (loss_coarse + loss_fine)
    return loss


optimizer = optim.Adam(model.parameters(), lr=1e-3)
train(model, optimizer, coarse_classes, fine_classes, num_steps=500)
