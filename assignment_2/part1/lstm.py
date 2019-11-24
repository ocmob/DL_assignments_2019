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

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu', save_grads=False):
        super(LSTM, self).__init__()
        self.device = device

        self.save_grads = save_grads

        # new activations come here
        self.wgx = nn.Parameter(torch.zeros(num_hidden, input_dim, device=device))
        nn.init.xavier_uniform_(self.wgx)
        self.wgh = nn.Parameter(torch.zeros(num_hidden, num_hidden, device=device))
        nn.init.xavier_uniform_(self.wgh)
        self.bg = nn.Parameter(torch.zeros(num_hidden, device=device))

        # gate for new activations
        self.wix = nn.Parameter(torch.zeros(num_hidden, input_dim, device=device))
        nn.init.xavier_uniform_(self.wgx)
        self.wih = nn.Parameter(torch.zeros(num_hidden, num_hidden, device=device))
        nn.init.xavier_uniform_(self.wgh)
        self.bi = nn.Parameter(torch.zeros(num_hidden, device=device))

        # gate for hidden state
        self.wfx = nn.Parameter(torch.zeros(num_hidden, input_dim, device=device))
        nn.init.xavier_uniform_(self.wgx)
        self.wfh = nn.Parameter(torch.zeros(num_hidden, num_hidden, device=device))
        nn.init.xavier_uniform_(self.wgh)
        self.bf = nn.Parameter(2*torch.ones(num_hidden, device=device))
        
        # output gate
        self.wox = nn.Parameter(torch.zeros(num_hidden, input_dim, device=device))
        nn.init.xavier_uniform_(self.wgx)
        self.woh = nn.Parameter(torch.zeros(num_hidden, num_hidden, device=device))
        nn.init.xavier_uniform_(self.wgh)
        self.bo = nn.Parameter(torch.zeros(num_hidden, device=device))

        # output layer and to softmax
        self.wph = nn.Parameter(torch.zeros(num_classes, num_hidden, device=device))
        nn.init.xavier_uniform_(self.wph)
        self.bp = nn.Parameter(torch.zeros(num_classes, device=device))



    def forward(self, x):

        hprev = torch.zeros(x.shape[0], self.wih.shape[0], device=self.device)
        cprev = torch.zeros(x.shape[0], self.wih.shape[0], device=self.device)

        if self.save_grads:
            self.grad_over_time = []

        for char_batch in x.T:

            char_batch = char_batch[:, None]
            #g = torch.tanh(char_batch @ self.wgx.T + hprev @ self.wgh + self.bg)
            g = char_batch @ self.wgx.T + hprev @ self.wgh + self.bg
            i = torch.sigmoid(char_batch @ self.wix.T + hprev @ self.wih + self.bi)
            f = torch.sigmoid(char_batch @ self.wfx.T + hprev @ self.wfh + self.bf)
            o = torch.sigmoid(char_batch @ self.wox.T + hprev @ self.woh + self.bo)
            cprev = g * i + cprev * f
            #hprev = torch.tanh(cprev) * o
            hprev = cprev * o

            if self.save_grads:
                self.grad_over_time.append(hprev)
                hprev.retain_grad()
                hprev = hprev.clone()

        p = hprev @ self.wph.T + self.bp

        return p
