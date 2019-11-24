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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

# TODO WTF?
#from part1.dataset import PalindromeDataset
#from part1.vanilla_rnn import VanillaRNN
#from part1.lstm import LSTM

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

def acc(predictions, targets, num_classes):

  with torch.no_grad():
      accuracy = (F.one_hot(predictions.max(dim=1).indices, 
          num_classes=num_classes).float() * targets).sum()/targets.sum()

  return accuracy.cpu().item()

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    test_loader = DataLoader(dataset, 10000, num_workers=1)

    if config.train_log != "STDOUT":
        outfile = open(config.train_log, 'w')

    accuracy_avg = 0
    iters = 1

    for i in range(iters):
        # Initialize the model that we are going to use
        if config.model_type == 'RNN':
            model = VanillaRNN(config.input_length, config.input_dim,
                    config.num_hidden, config.num_classes, device)
            optimizer = optim.SGD(model.parameters(), config.learning_rate)
        else:
            model = LSTM(config.input_length, config.input_dim,
                    config.num_hidden, config.num_classes, device)
            optimizer = optim.RMSprop(model.parameters(), config.learning_rate)

        model.to(device)

        # Setup the loss and optimizer
        criterion = nn.CrossEntropyLoss()  

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            pred = model.forward(batch_inputs)

            ############################################################################
            # QUESTION: what happens here and why?
            # Gradient clipping is performed. In deep computational graphs the 
            # parameter gradient could grow very large due to a large number of 
            # repeatedly applying the same operation. If this happens an SGD 
            # update will take a bigger-than-usual step, possibly ending up in a 
            # region where loss function already begins to curve upwards again.
            # To alleviate this behaviour we perform gradient clipping, which
            # restricts the maximum possible value of gradient and thus the max step
            # we can take. This will make convergence easier and the optimization 
            # process will be better-behaved than without gradient clipping
            ############################################################################
            # Clipping moved to AFTER the backward pass
            ############################################################################

            loss = criterion(pred, batch_targets)
            accuracy = acc(pred, F.one_hot(batch_targets, num_classes=config.num_classes).float(), config.num_classes) 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            if step % 10 == 0:
                if config.train_log != "STDOUT":
                    outfile.write("[{}] Averaging Step: {} Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                          "Accuracy = {:.2f}, Loss = {:.3f}\n".format(
                            datetime.now().strftime("%Y-%m-%d %H:%M"), i, step,
                            config.train_steps, config.batch_size, examples_per_second,
                            accuracy, loss
                    ))
                else:
                    print("[{}] Averaging Step: {} Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                          "Accuracy = {:.2f}, Loss = {:.3f}".format(
                            datetime.now().strftime("%Y-%m-%d %H:%M"), i, step,
                            config.train_steps, config.batch_size, examples_per_second,
                            accuracy, loss
                    ))

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break

        test_inputs, test_targets = next(iter(test_loader))
        test_inputs = test_inputs.to(device)
        test_targets = test_targets.to(device)

        with torch.no_grad():
            pred = model.forward(test_inputs)
            loss = criterion(pred, test_targets)

        accuracy = acc(pred, F.one_hot(test_targets, num_classes=config.num_classes).float(), config.num_classes) 
        accuracy_avg += accuracy 

    print(accuracy_avg/iters, end='')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--linear', action='store_true', help="Make the net linear")
    parser.add_argument('--train_log', type=str, default="STDOUT", help="Output file name")

    config = parser.parse_args()

    # Train the model
    train(config)
