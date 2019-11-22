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
import torch.nn.functional as F

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()

        self.seq_length = seq_length

        # Initialization here ...
        self.wph = nn.Parameter(torch.zeros(num_classes, num_hidden)).to(device)
        nn.init.xavier_uniform_(self.wph)
        self.whh = nn.Parameter(torch.zeros(num_hidden, num_hidden)).to(device)
        nn.init.xavier_uniform_(self.whh)
        self.whx = nn.Parameter(torch.zeros(num_hidden, input_dim)).to(device)
        nn.init.xavier_uniform_(self.whx)

        self.bh = nn.Parameter(torch.zeros(num_hidden)).to(device)
        self.bp = nn.Parameter(torch.zeros(num_classes)).to(device)

    def forward(self, x):
        # Implementation here ...
        hprev = torch.zeros(x.shape[0], self.whh.shape[0])
        for char_batch in x.T:
            hprev = F.tanh(char_batch[:, None] @ self.whx.T + hprev @ self.whh.T + self.bh)
        p = hprev @ self.wph.T + self.bp

        return p
