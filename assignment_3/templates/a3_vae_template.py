import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from datasets.bmnist import bmnist

class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, data_dim=28*28, deep=False):
        super().__init__()
        self.hid_lin = torch.nn.Linear(data_dim, hidden_dim)
        self.hid_act = torch.nn.ReLU()

        if deep:
            self.deep_lin = torch.nn.Linear(hidden_dim, hidden_dim)
            self.deep_act = torch.nn.ReLU()

        self.sigma_lin = torch.nn.Linear(hidden_dim, z_dim)
        self.mu_lin = torch.nn.Linear(hidden_dim, z_dim)


    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """

        # TODO CONSTRAINTS?
        x = self.hid_lin.forward(input)
        x = self.hid_act.forward(x)

        if deep:
            x = self.deep_lin.forward(x)
            x = self.deep_act.forward(x)

        mean = self.mu_lin.forward(x)
        std = self.sigma_lin.forward(x)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, data_dim=28*28, deep=False):
        super().__init__()
        self.hid_lin = torch.nn.Linear(z_dim, hidden_dim)
        self.hid_act = torch.nn.ReLU()

        if deep:
            self.deep_lin = torch.nn.Linear(hidden_dim, hidden_dim)
            self.deep_act = torch.nn.ReLU()

        self.out_lin = torch.nn.Linear(hidden_dim, data_dim)
        self.out_act = torch.nn.Sigmoid()

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """
        x = self.hid_lin.forward(input)
        x = self.hid_act.forward(x)

        if deep:
            x = self.deep_lin.forward(x)
            x = self.deep_act.forward(x)

        x = self.out_lin.forward(x)
        mean = self.out_act.forward(x)

        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20, deep=False):
        super().__init__()

        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim, deep)
        self.decoder = Decoder(hidden_dim, z_dim, deep)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """

        # TODO ANY LOG TRICKS HERE?
        mu, logsig = self.encoder.forward(input)
        sample = torch.normal(torch.zeros_like(mu), 
                torch.ones_like(logsig))

        z = mu + torch.sqrt(torch.exp(logsig))*sample
        mu_out = self.decoder.forward(z)

        kl = -1/2*(1+logsig-mu.pow(2)-torch.exp(logsig)).sum(dim=1)
        logp = -(input*torch.log(mu_out) + (1-input)*torch.log(1-mu_out)).sum(dim=1)

        # TODO NEGATIVE?
        average_negative_elbo = (kl+logp).mean()

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """

        with torch.no_grad():
            samples = torch.normal(torch.zeros(n_samples, self.z_dim), 
                    torch.ones(n_samples, self.z_dim))

            im_means = self.decoder.forward(samples)
            sampled_ims = torch.bernoulli(im_means)
        
        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    
    average_epoch_elbo = 0
    if ARGS.t:
        dataiter = iter(data)
        for i in range(5):
            images = next(dataiter)
            optimizer.zero_grad()
            elbo_iter = model.forward(images.view(images.shape[0],-1))
            if not elbo_iter.requires_grad:
                breakpoint()
            elbo_iter.backward()
            average_epoch_elbo += elbo_iter.item()
            del elbo_iter
            optimizer.step()
    else:
        for i, images in enumerate(data):
            optimizer.zero_grad()
            elbo_iter = model.forward(images.view(images.shape[0],-1))
            if not elbo_iter.requires_grad:
                breakpoint()
            elbo_iter.backward()
            average_epoch_elbo += elbo_iter.item()
            del elbo_iter
            optimizer.step()

    average_epoch_elbo /= i+1

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    #TODO nograd
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def main():
    if ARGS.t:
        data = bmnist(batch_size=1)[:2]  # ignore test split
    else:
        data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim, True)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 40)

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")
        scheduler.step()

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        if ((epoch+1) % 5) == 0:
            print("[Epoch {}], current learning rate: {}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
            imgs, img_means = model.sample(10)
            grid = make_grid(imgs.view(10, 1, 28, -1), nrow = 2)
            import matplotlib.pyplot as plt
            plt.imshow(grid.permute(2, 1, 0).numpy())
            plt.title("Samples epoch = {}".format(epoch+1))
            plt.savefig("./results/samples_epoch_{}.pdf".format(epoch+1))


    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('-t', action='store_true',
                        help='run in test mode')

    ARGS = parser.parse_args()

    main()
