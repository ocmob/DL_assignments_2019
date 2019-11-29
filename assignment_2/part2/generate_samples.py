# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from part2.dataset import TextDataset
from part2.model import TextGenerationModel

################################################################################

def sample(config):

    dataset = TextDataset(config.base_txt, config.seq_length)

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    model = torch.load(config.model_file, map_location=device)

    if config.samples_out_file != "STDOUT":
        samples_out_file = open(config.samples_out_file, 'w')

    with torch.no_grad():
        codes = []

        print(dataset.vocab_size)

        for i in range(config.no_random):
            input_tensor = torch.zeros((1, 1, dataset.vocab_size), device=device)
            input_tensor[0, 0, np.random.randint(0, dataset.vocab_size)] = 1

            for i in range(config.seq_length-1):
                response = model.step(input_tensor)
                logits = F.log_softmax(config.temp*response, dim=1)
                dist = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)
                code = dist.sample().argmax().item()
                input_tensor *= 0
                input_tensor[0, 0, code] = 1
                codes.append(code)

            string = dataset.convert_to_string(codes)
            model.reset_stepper()

            if config.samples_out_file != "STDOUT":
                samples_out_file.write(string + "\n")
            else:
                print(string)

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--base_txt', type=str, required=True, help="Path to an .txt file used to train the model")
    parser.add_argument('--model_file', type=str, required=True, help="Path to an .out file with a saved, trained model")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an output sequence')
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--temp', type=float, default=1.0, help="Temperature parameter for random sampling")
    parser.add_argument('--samples_out_file', type=str, default="STDOUT", help="Output file with a train model. Default: print to STDOUT")
    parser.add_argument('--beam_width', type=int, default=5, help='Width of a beam in beam search')

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--sentence', type=str, help='Input string to be completed')
    group.add_argument('--no_random', type=int, help='Number of output samples starting with random letters')

    config = parser.parse_args()

    # Train the model
    sample(config)
