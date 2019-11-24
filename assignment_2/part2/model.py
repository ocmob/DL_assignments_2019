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

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        self.lstm = nn.LSTM(input_size = 1,
                hidden_size = lstm_num_hidden,
                num_layers = lstm_num_layers).to(device)

        self.module_list = nn.ModuleList()
        for i in range(seq_length):
            self.module_list.append(nn.Linear(lstm_num_hidden, vocabulary_size).to(device))

    def forward(self, x):

        x = x.T[:,:, None].float()
        lstmout, _ = self.lstm(x)
        for i, tensor in enumerate(lstmout):
            if i == 0:
                out = self.module_list[i](tensor)[:,:,None]
            else:
                out = torch.cat((out, self.module_list[i](tensor)[:,:,None]), 2)

        return out

    def reset_stepper(self):
        self.stepper = 0

    def step(self, x):
        x = x.T[:,:, None].float()
        lstmout, _ = self.lstm(x)
        out = self.module_list[self.stepper](lstmout[0])

        return out
