import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from datasets.mnist import mnist
import os
from torchvision.utils import make_grid

device = torch.device('cuda:0')

def log_prior(x):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """
    halflog2pi = 0.399
    logp = torch.diag(-x.shape[1]*halflog2pi-1/2*(x @ x.T))

    ## ===> PREVIOUS
    #pi_tensor = torch.tensor(2*np.pi)
    #logp = torch.sum(- 0.5 * x.pow(2) - torch.log(torch.sqrt(pi_tensor)), dim=1)

    return logp


def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """

    sample = torch.normal(torch.ones(size), torch.zeros(size))

    ## ===> PREVIOUS
    #sample = torch.randn(size)

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
            nn.Linear(n_hidden, 2*c_in)
            )
#
#        # The nn should be initialized such that the weights of the last layer
#        # is zero, so that its initial transform is identity.
#        self.nn[-1].weight.data.zero_()
#        self.nn[-1].bias.data.zero_()
#        self.nn = torch.nn.Sequential(
#            nn.Linear(c_in, n_hidden),
#            nn.ReLU(),
#            nn.Linear(n_hidden, n_hidden),
#            nn.ReLU(),
#            )
#        self.trans_net = nn.Linear(n_hidden, c_in)
#        self.scale_net = nn.Sequential(
#            nn.Linear(n_hidden, c_in),
#            nn.Tanh()
#        )

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()
#        self.trans_net.weight.data.zero_()
#        self.trans_net.bias.data.zero_()
#        self.scale_net[0].weight.data.zero_()
#        self.scale_net[0].bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.

        #if not reverse:
        #    nnout = self.nn.forward(self.mask * z)
        #    logscale = torch.tanh(nnout)
        #    z = self.mask*z + (1-self.mask)*(z*torch.exp(logscale) + nnout)
        #    ldj += ((1-self.mask)*logscale).sum(dim = 1)
        #else:
        #    nnout = self.nn.forward(self.mask * z)
        #    logscale = -torch.tanh(nnout)
        #    z = self.mask*z + (1-self.mask)*(z - nnout)*torch.exp(logscale)
        #    ldj += ((1-self.mask)*logscale).sum(dim = 1)
        z_masked = z * self.mask
        nnout = self.nn(z_masked)
        log_scale, trans = torch.chunk(nnout, 2, dim = 1)
        log_scale = torch.tanh(log_scale)

        #log_scale, trans = self.nn(z_masked).chunk(2, dim=1)
        #log_scale = self.tanh(log_scale)


        if not reverse:
            #straight direction
            z = z_masked + (1 - self.mask) * (z * torch.exp(log_scale) + trans)
            #compute log determinant Jacobian
            ldj += torch.sum((1 - self.mask) * log_scale, dim=1)
        else:
            #inverse direction
            z = z_masked + (1 - self.mask) * (z - trans) * torch.exp(-log_scale)
            #set to zero
            ldj = torch.zeros_like(ldj)


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

    def debug(self, input):
        with torch.no_grad():
            # Forward
            z = input

            grid = make_grid(
                    z.cpu().view(-1, 1, 28, 28).permute(0, 1, 3, 2), 
                    nrow = 5, normalize=True)
            plt.imshow(grid.permute(2, 1, 0).numpy())
            plt.show()

            ldj = torch.zeros(z.size(0), device=z.device)
            z = self.dequantize(z)
            z, ldj = self.logit_normalize(z, ldj)
            z, ldj = self.flow(z, ldj)

            # Backward
            z, _ = self.flow(z, ldj, reverse=True)
            z, _ = self.logit_normalize(z, _, reverse=True)

            grid = make_grid(
                    z.cpu().view(-1, 1, 28, 28).permute(0, 1, 3, 2), 
                    nrow = 5, normalize=True)
            plt.imshow(grid.permute(2, 1, 0).numpy())
            plt.show()


    def forward(self, input):
        #"""
        #Given input, encode the input to z space. Also keep track of ldj.
        #"""
        
        #z = input
        #ldj = torch.zeros(z.size(0), device=z.device)
        #z = self.dequantize(z)
        #z, ldj = self.logit_normalize(z, ldj)
        #z, ldj = self.flow(z, ldj)
        #log_px = log_prior(z) + ldj
        z = input
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)

        # Compute log_pz and log_px per example
        log_pz = log_prior(z)
        log_px = log_pz + ldj

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """

        #with torch.no_grad():
        #    z = sample_prior((n_samples,) + self.flow.z_shape)
        #    ldj = torch.zeros(z.size(0), device=z.device)
        #    z, ldj = self.flow(z, ldj, reverse=True)
        #    z, _ = self.logit_normalize(z, ldj, reverse=True)
        z = sample_prior((n_samples,) + self.flow.z_shape).to(device)
        ldj = torch.zeros(z.size(0), device=z.device)

        #compute reverse flow
        z, ldj = self.flow.forward(z, ldj, reverse=True)
        z, _ = self.logit_normalize(z, ldj, reverse=True)

        return z


def epoch_iter(model, data, optimizer, test_mode = False, 
        device = torch.device('cuda:0')):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """

    bpds = 0.0

    for i, (imgs, _) in enumerate (data):

        #forward pass
        imgs = imgs.to(device)
        log_px = model.forward(imgs)

        #compute loss
        loss = -1*torch.mean(log_px)
        bpds += loss.item()

        #backward pass only when in training mode
        if model.training:
            optimizer.zero_grad()
            loss.backward()

            #clip gradient to avoid exploding grads
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

        #if i == 100:
            #break

        #print(bpds/(28**2*np.log(2)))
        #os.exit()

    #for readibility
    n_batches = i + 1
    img_shape = imgs.shape[1] # 28 x 28
    #compute average bit per dimension for one epoch
    avg_bpd = bpds / (n_batches * (28**2) * np.log(2))

    return avg_bpd

    #avg_bpd = 0
    #dataiter = iter(data)

    #if test_mode: 
    #    iters = 5
    #else:
    #    iters = len(dataiter)

    #log2 = 0.30103

    #if model.training:
    #    for i in range(iters):
    #        optimizer.zero_grad()
    #        imgs, _ = next(dataiter)
    #        imgs = imgs.to(device).reshape(-1, 28*28)
    #        log_px = -model.forward(imgs).mean()
    #        log_px.backward()
    #        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    #        optimizer.step()

    #        avg_bpd += (log_px/log2).item()
    #else:
    #    with torch.no_grad():
    #        for i in range(iters):
    #            optimizer.zero_grad()
    #            imgs, _ = next(dataiter)
    #            imgs = imgs.to(device).reshape(-1, 28*28)
    #            log_px = -model(imgs).mean()

    #            avg_bpd += (log_px/log2).item()
    #
    #return avg_bpd/(i+1)/28**2


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

def save_nf_samples(model, epoch, path, n_row=5):
    n_samples = n_row**2
    with torch.no_grad():
        #sample from model
        sample_ims = model.sample(n_samples).detach()
        #reshape
        sample_ims = sample_ims.reshape(n_samples, 1, 28, 28)

        #transform samples to grid represenation
        sample_ims = make_grid(sample_ims, nrow=n_row, normalize=True)

        #format sampled images
        samples = sample_ims.cpu().numpy().transpose(1,2,0)

        #save samples
        file_name = (f"sample_{epoch}.png")
        plt.imsave(path + file_name, samples)
        print(f"Saved {file_name}\n")


def main():
    if(ARGS.t):
        data = mnist(root=ARGS.dpath, batch_size=16
                )[:2]  # ignore test split
    else:
        data = mnist(root=ARGS.dpath)[:2]  # ignore test split

    device = torch.device(ARGS.device)

    model = Model(shape=[784])
    model.to(device)

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
        #save_nf_samples(model, epoch, 'images_nfs/')
        os.makedirs('images_nfs/', exist_ok=True)

        NO_IMAGES = 25
        samples = model.sample(NO_IMAGES)
        grid = make_grid(
                samples.cpu().view(NO_IMAGES, 1, 28, -1).permute(0, 1, 3, 2), 
                nrow = 5, normalize=True)
        plt.imshow(grid.permute(2, 1, 0).detach().numpy())
        plt.title('Sample generated image, epoch {}'.format(epoch))
        plt.savefig('images_nfs/epoch_{}.png'.format(epoch))
    #data = mnist(batch_size=16)[:2]  # ignore test split

    #res_path = './images_nfs/'


    ##initialise model
    #model = Model(shape=[784])
    #model.to(device)

    ##initialise Adam optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    #os.makedirs(res_path, exist_ok=True)

    #train_curve, val_curve = [], []
    #import time
    #for epoch in range(ARGS.epochs):

    #    t0 = time.time()
    #    bpds = run_epoch(model, data, optimizer, False, device)
    #    t1 = time.time()
    #    train_bpd, val_bpd = bpds
    #    train_curve.append(train_bpd)
    #    val_curve.append(val_bpd)
    #    print(f"[Epoch {epoch+1}/{ARGS.epochs}] train bpd: {train_bpd:.3f} val_bpd: {val_bpd:.3f} time: {t1-t0:.2f}s")

    #    # --------------------------------------------------------------------
    #    #  Add functionality to plot samples from model during training.
    #    #  You can use the make_grid functionality that is already imported.
    #    #  Save grid to images_nfs/
    #    # --------------------------------------------------------------------
    #    # similar to function used for VAEs
    #    save_nf_samples(model, epoch, res_path)


    #    #save intermediate results
    #    #losses = {"train_loss": train_curve, "val_loss": val_curve}
    #    #np.save(res_path + "train_val_loss", losses)

    #    #torch.save(model.state_dict(), res_path + "nf_model.pt")

    save_bpd_plot(train_curve, val_curve, 'nfs_bpd.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('-t', action='store_true',
                        help='run in test mode')
    parser.add_argument('--device', type=str, default="cuda:0", 
            help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--dpath', type=str, default='./data/',
                        help='Root path for dataset')

    ARGS = parser.parse_args()

    main()
