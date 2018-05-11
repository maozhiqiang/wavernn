#!/usr/bin/env python
#-*- coding:utf-8 -*-

class Hyperparams:
    '''Hyper parameters'''
    # pipeline
    # signal processing
    sr = 16000  # Sampling rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
    n_mels = 80  # Number of Mel banks to generate
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97
    max_db = 100
    ref_db = 20

    # Model


    # training scheme
    lr = 0.001 # Initial learning rate.
    B = 32 #32 # batch size
    num_iterations = 2000000

