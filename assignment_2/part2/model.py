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

        self.stepper = 0
        self.lstm = nn.LSTM(input_size = vocabulary_size,
                hidden_size = lstm_num_hidden,
                num_layers = lstm_num_layers,
                batch_first=True).to(device)
        self.linear=nn.Linear(lstm_num_hidden, vocabulary_size).to(device)

    def forward(self, x):
        lstmout, _ = self.lstm(x, None)
        out = self.linear(lstmout)

        return out

    def reset_stepper(self):
        self.stepper = 0
        del self.step_hidden

    def step(self, x):
        if self.stepper == 0:
            lstmout, self.step_hidden = self.lstm(x)
        else:
            lstmout, self.step_hidden = self.lstm(x, self.step_hidden)
        out = self.linear(lstmout[0])
        self.stepper += 1

        return out
