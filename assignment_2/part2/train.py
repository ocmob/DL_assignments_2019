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

def acc(predictions, targets, num_classes):

  with torch.no_grad():
      accuracy = (F.one_hot(predictions.max(dim=1).indices, num_classes=num_classes).float() * targets).sum()/targets.sum()

  return accuracy.detach().cpu().item()

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length,
            dataset.vocab_size, config.lstm_num_hidden, config.lstm_num_layers,
            device)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.learning_rate_step, gamma=config.learning_rate_decay)

    accuracy_train = []
    loss_train = []

    if config.samples_out_file != "STDOUT":
        samples_out_file = open(config.samples_out_file, 'w')

    epochs = config.train_steps // len(data_loader) + 1

    print("Will train on {} batches in {} epochs, max {} batches/epoch.".format(config.train_steps, epochs, len(data_loader)))

    for epoch in range(epochs):
        data_loader_iter = iter(data_loader)

        if epoch == config.train_steps // len(data_loader):
            batches = config.train_steps % len(data_loader)
        else:
            batches = len(data_loader)

        for step in range(batches):
            batch_inputs, batch_targets = next(data_loader_iter)
            model.zero_grad()

            # Only for time measurement of step through network
            t1 = time.time()

            batch_inputs = F.one_hot(batch_inputs, num_classes=dataset.vocab_size,
                    ).float().to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()

            pred, _ = model.forward(batch_inputs)
            loss = criterion(pred.transpose(2, 1), batch_targets)
            accuracy = acc(pred.transpose(2, 1), F.one_hot(batch_targets, num_classes=dataset.vocab_size).float(), dataset.vocab_size) 
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optimizer.step()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            scheduler.step()

            if (epoch*len(data_loader) + step + 1) % config.seval_every == 0:
                accuracy_train.append(accuracy)
                loss_train.append(loss.item())

            if (epoch*len(data_loader) + step + 1) % config.print_every == 0:
                print("[{}] Epoch: {:04d}/{:04d}, Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), epoch+1, epochs, (epoch*len(data_loader) + step + 1),
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                ))

            if (epoch*len(data_loader) + step + 1) % config.sample_every == 0:
                with torch.no_grad():
                    codes = []

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
                        samples_out_file.write("Step {}: ".format(epoch*len(data_loader) + step + 1) + string + "\n")
                    else:
                        print(string)


    if config.samples_out_file != "STDOUT":
        samples_out_file.close()

    if config.model_out_file != None:
        torch.save(model, config.model_out_file)
        
    if config.curves_out_file != None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        fig.suptitle('Training curves for Pytorch 2-layer LSTM.\nFinal loss: {:.4f}. Final accuracy: {:.4f}\nSequence length: {}, Hidden units: {}, LSTM layers: {}, Learning rate: {:.4f}'.format(
            loss_train[-1], accuracy_train[-1], config.seq_length, config.lstm_num_hidden, config.lstm_num_layers, config.learning_rate))
        plt.subplots_adjust(top=0.8)

        ax[0].set_title('Loss')
        ax[0].set_ylabel('Loss value')
        ax[0].set_xlabel('No of batches seen x{}'.format(config.seval_every))
        ax[0].plot(loss_train, label='Train')
        ax[0].legend()

        ax[1].set_title('Accuracy')
        ax[1].set_ylabel('Accuracy value')
        ax[1].set_xlabel('No of batches seen x{}'.format(config.seval_every))
        ax[1].plot(accuracy_train, label='Train')
        ax[1].legend()
        
        plt.savefig(config.curves_out_file)

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--train_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--seval_every', type=int, default=100, help='How often to save metrics from the model')

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--temp', type=float, default=1.0, help="Temperature parameter for random sampling")

    parser.add_argument('--samples_out_file', type=str, default="STDOUT", help="Output file with a train model. Default: print to STDOUT")
    parser.add_argument('--model_out_file', type=str, default=None, help="Output file with a train model. Default: do not save")
    parser.add_argument('--curves_out_file', type=str, default=None, help="Output file with training curve plots. Default: do not save")

    config = parser.parse_args()

    # Train the model
    train(config)
