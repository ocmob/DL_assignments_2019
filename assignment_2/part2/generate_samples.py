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

    if config.no_random != None:
        with torch.no_grad():
            codes = []

            for k in range(config.no_random):
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
                    samples_out_file.write("Sample {}: ".format(k)+string + "\n")
                else:
                    print("Sample {}: ".format(k)+string)
                string = ''
                codes = []
    elif config.sentence != None:
        with torch.no_grad():
            codes = []
            for char in config.sentence:
                codes.append(dataset._char_to_ix[char])
            input_tensor = torch.zeros((1, len(codes), dataset.vocab_size), device=device)
            input_tensor[0, np.arange(0, len(codes), 1), codes] = 1

            chars_to_gen = config.seq_length - len(codes)

            for i in range(len(codes)):
                response = model.step(input_tensor[:, i, :].view(1, 1, dataset.vocab_size))

            input_tensor = torch.zeros((1, 1, dataset.vocab_size), device=device)
            for i in range(chars_to_gen):
                logits = F.log_softmax(config.temp*response, dim=1)
                dist = torch.distributions.one_hot_categorical.OneHotCategorical(logits=logits)
                code = dist.sample().argmax().item()
                input_tensor *= 0
                input_tensor[0, 0, code] = 1
                codes.append(code)
                response = model.step(input_tensor)

            string = dataset.convert_to_string(codes)
            model.reset_stepper()

            if config.samples_out_file != "STDOUT":
                samples_out_file.write(string + "\n")
            else:
                print(string)
    else:
        with torch.no_grad():
            codes = []
            beams = []
            for k in range(config.beam_width):
                beam_dict = {}
                beam_dict['hidden_state'] = None
                beam_dict['logit'] = -np.log(config.beam_width)
                beam_dict['seq_codes'] = [np.random.randint(0, dataset.vocab_size)]
                beams.append(beam_dict)

            input_tensor = torch.zeros((1, 1, dataset.vocab_size), device=device)

            import copy

            for i in range(config.seq_length):
                new_beams = []

                for element in beams:
                    input_tensor *= 0
                    input_tensor[0,0,element['seq_codes'][-1]] = 1.0
                    response, hid = model.forward(input_tensor, element['hidden_state'])
                    logits = F.log_softmax(config.temp*response, dim=2)
                    for code, logit in enumerate(logits[0,0,:]):
                        new_dict = copy.deepcopy(element)
                        new_dict['hidden_state'] = hid
                        new_dict['seq_codes'].append(code)
                        new_dict['logit'] += logit.item()
                        new_beams.append(new_dict)
                new_beams.sort(reverse=True, key=lambda dic: dic['logit'])
                beams = new_beams[:config.beam_width]
            for beam in beams:
                string = dataset.convert_to_string(beam['seq_codes'])
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

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--sentence', type=str, default=None, help='Input string to be completed')
    group.add_argument('--no_random', type=int, default=None, help='Number of output samples starting with random letters')
    group.add_argument('--beam_width', type=int, default=None, help='Width of a beam in beam search')

    config = parser.parse_args()

    # Train the model
    sample(config)
