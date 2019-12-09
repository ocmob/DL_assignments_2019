import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from datasets.mnist import mnist
import os
from torchvision.utils import make_grid


def log_prior(x):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """
    
    logp = torch.diag(-x.shape[1]/1.5962-1/2*(x @ x.T))

    return logp


def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """

    sample = torch.normal(torch.ones(size), torch.zeros(size))

    if torch.cuda.is_available():
        sample = sample.cuda()

    return sample


def get_mask():
    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28*28)
    mask = torch.from_numpy(mask)

    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024):
        super().__init__()
        self.n_hidden = n_hidden

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.
        # Suggestion: Linear ReLU Linear ReLU Linear.
        self.nn = torch.nn.Sequential(
            nn.Linear(c_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, c_in)
            )

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.

        if not reverse:
            nnout = self.nn.forward(self.mask * z)
            logscale = torch.tanh(nnout)
            z = self.mask*z + (1-self.mask)*(z*torch.exp(logscale) + nnout)
            ldj += logscale.sum(dim = 1)
        else:
            nnout = self.nn.forward(self.mask * z)
            logscale = -torch.tanh(nnout)
            z = self.mask*z + (1-self.mask)*(z - nnout)*torch.exp(logscale)
            ldj += logscale.sum(dim = 1)

        return z, ldj


class Flow(nn.Module):
    def __init__(self, shape, n_flows=4):
        super().__init__()
        channels, = shape

        mask = get_mask()

        self.layers = torch.nn.ModuleList()

        for i in range(n_flows):
            self.layers.append(Coupling(c_in=channels, mask=mask))
            self.layers.append(Coupling(c_in=channels, mask=1-mask))

        self.z_shape = (channels,)

    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z, logdet


class Model(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.flow = Flow(shape)

    def dequantize(self, z):
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            # Divide by 256 and update ldj.
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z*(1-alpha) + alpha*0.5
            logdet += torch.sum(-torch.log(z) - torch.log(1-z), dim=1)
            z = torch.log(z) - torch.log(1-z)

        else:
            # Inverse normalize
            z = torch.sigmoid(z)
            logdet += torch.sum(torch.log(z) + torch.log(1-z), dim=1)
            z = (z - alpha*0.5)/(1 - alpha)

            # Multiply by 256.
            logdet += np.log(256) * np.prod(z.size()[1:])
            z = z * 256.

        return z, logdet

    def forward(self, input):
        """
        Given input, encode the input to z space. Also keep track of ldj.
        """
        z = input
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)
        #z_sig = torch.nn.functional.sigmoid(z)
        #ldj_sig = torch.nn.functional.sigmoid(ldj)

        ## Compute log_pz and log_px per example
        ## TODO ?? sum
        #if torch.isnan(log_px).any():
        #    breakpoint()

        log_px = log_prior(z) + ldj.view(z.shape[0],-1)

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """

        #TODO logit normalization?
        z = sample_prior((n_samples,) + self.flow.z_shape)
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, _ = self.flow(z, ldj, reverse=True)

        return z


def epoch_iter(model, data, optimizer, test_mode = False, 
        device = torch.device('cuda:0')):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """

    avg_bpd = 0
    
    dataiter = iter(data)

    if test_mode: 
        iters = 5
    else:
        iters = len(dataiter)

    if model.training:
        for i in range(iters):
            optimizer.zero_grad()
            imgs, _ = next(dataiter)
            imgs = imgs.to(device).reshape(-1, 28*28)
            log_px = -model(imgs).sum()
            log_px.backward()
            optimizer.step()

            avg_bpd += (log_px/0.30103).item()
    else:
        with torch.no_grad():
            for i in range(iters):
                optimizer.zero_grad()
                imgs, _ = next(dataiter)
                imgs = imgs.to(device).reshape(-1, 28*28)
                log_px = -model(imgs).sum()

                avg_bpd += (log_px/0.30103).item()
    
    return avg_bpd/(i+1)


def run_epoch(model, data, optimizer, test_mode = False, 
        device = torch.device('cuda:0')):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer, test_mode, device)

    model.eval()
    val_bpd = epoch_iter(model, valdata, optimizer, test_mode, device)

    return train_bpd, val_bpd


def save_bpd_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train bpd')
    plt.plot(val_curve, label='validation bpd')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('bpd')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    if(ARGS.t):
        data = mnist(batch_size=2)[:2]  # ignore test split
    else:
        data = mnist()[:2]  # ignore test split

    model = Model(shape=[784])

    device = torch.device(ARGS.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    os.makedirs('images_nfs', exist_ok=True)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        bpds = run_epoch(model, data, optimizer, ARGS.t, device)
        train_bpd, val_bpd = bpds
        train_curve.append(train_bpd)
        val_curve.append(val_bpd)
        print("[Epoch {epoch}] train bpd: {train_bpd} val_bpd: {val_bpd}".format(
            epoch=epoch, train_bpd=train_bpd, val_bpd=val_bpd))

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        #  Save grid to images_nfs/
        # --------------------------------------------------------------------

    save_bpd_plot(train_curve, val_curve, 'nfs_bpd.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('-t', action='store_true',
                        help='run in test mode')
    parser.add_argument('--device', type=str, default="cuda:0", 
            help="Training device 'cpu' or 'cuda:0'")

    ARGS = parser.parse_args()

    main()
