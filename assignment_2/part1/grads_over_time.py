################################################################################
# MIT License
# 
# Copyright (c) 2019
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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

from part1.dataset import PalindromeDataset
from part1.vanilla_rnn import VanillaRNN
from part1.lstm import LSTM

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

def acc(predictions, targets, num_classes):

  with torch.no_grad():
      accuracy = (F.one_hot(predictions.max(dim=1).indices, num_classes=num_classes).float() * targets).sum()/targets.sum()

  return accuracy.detach().cpu().item()

def get_grads(model, device, data_loader, dataset, num_classes):
    model.to(device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()  

    batch_inputs, batch_targets = next(iter(data_loader))

    batch_inputs = batch_inputs.to(device)
    batch_targets = batch_targets.to(device)

    pred = model.forward(batch_inputs)

    loss = criterion(pred, batch_targets)
    accuracy = acc(pred, F.one_hot(batch_targets, num_classes=num_classes).float(), num_classes) 

    loss.backward()
    
    grad_norms = []
    for hid_state in model.grad_over_time:
        grad_norms.append(hid_state.grad.abs().sum().item())
    
    return grad_norms

def train(config):

    # Initialize the device which to run the model on
    device = torch.device("cpu")

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, 1, num_workers=1)

    # Initialize the model that we are going to use
    model = VanillaRNN(config.input_length, config.input_dim,
            config.num_hidden, config.num_classes, device, True, False)
    grads_vanilla = get_grads(model, device, data_loader, dataset, config.num_classes)

    model = LSTM(config.input_length, config.input_dim,
            config.num_hidden, config.num_classes, device, True, False)
    grads_lstm = get_grads(model, device, data_loader, dataset, config.num_classes)


    import matplotlib.pyplot as plt
    plt.plot(grads_vanilla, "o", label='VanillaRNN')
    plt.plot(grads_lstm, "o", label='LSTM')
    plt.yscale("log")
    plt.xlabel("Time step")
    plt.ylabel("Gradient magnitude (log)")
    plt.title("Comparison of gradient backprop in types of RNN\nInitialization: xavier_uniform, LSTM forget gate bias: 2")
    plt.legend()
    plt.show()


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=50, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')

    config = parser.parse_args()

    # Train the model
    train(config)
