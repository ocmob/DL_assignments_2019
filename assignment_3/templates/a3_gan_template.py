import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision import datasets


class Generator(nn.Module):
    def __init__(self, latent_dim=100, neg_slope = 0.2):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 784
        #   Output non-linearity

        self.mod_list = torch.nn.ModuleList()

        self.mod_list.append(nn.Linear(latent_dim, 128))
        self.mod_list.append(nn.LeakyReLU(neg_slope))
        for i in range(1,4):
            self.mod_list.append(nn.Linear(128*i, 128*(1+i)))
            self.mod_list.append(nn.BatchNorm1d(128*(1+i)))
            self.mod_list.append(nn.LeakyReLU(neg_slope))
        self.mod_list.append(nn.Linear(128*(1+i), 784))

    def forward(self, z):
        for module in self.mod_list:
            z = module(z)
        return z


class Discriminator(nn.Module):
    def __init__(self, latent_dim=100, neg_slope = 0.2):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity
        
        self.mod_list = torch.nn.ModuleList()

        self.mod_list.append(nn.Linear(784, 512))
        self.mod_list.append(nn.LeakyReLU(neg_slope))
        self.mod_list.append(nn.Linear(512, 256))
        self.mod_list.append(nn.LeakyReLU(neg_slope))
        self.mod_list.append(nn.Linear(256, 1))
        self.mod_list.append(nn.Sigmoid())

    def forward(self, img):
        for module in self.mod_list:
            img = module(img)
        return img

def gen_imgs(generator, latent_dim, device, no_images, title, filename):
    with torch.no_grad():
        no_images = 25
        zbatch = torch.normal(torch.zeros(no_images, latent_dim), 
                torch.ones(no_images, latent_dim)).to(device)
        fakebatch = generator(zbatch)
        grid = make_grid(
                fakebatch.cpu().view(no_images, 1, 28, -1).permute(0, 1, 3, 2), 
                nrow = 5, normalize = True)
        plt.imshow(grid.permute(2, 1, 0).numpy())
        plt.title(title)
        plt.savefig(filename)



def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, 
        device = torch.device('cuda:0'), img_dir=None, test_mode=False,
        d_steps=1, g_steps=1, latent_dim=100, scheduler_G=None, scheduler_D=None, 
        bsize=None):
    EPS = 1e-10

    for epoch in range(args.n_epochs):
        d_loss = 0
        g_loss = 0
        batches_done = 0

        dataiter = iter(dataloader)

        if test_mode:
            no_iters = 5
        else:
            no_iters = len(dataiter)

        for i in range(no_iters):
            imgs, _ = next(dataiter)
            imgs = imgs.to(device).reshape(-1, 28*28)
            for d in range(d_steps):
                optimizer_D.zero_grad()
                zbatch = torch.normal(torch.zeros(imgs.shape[0], latent_dim),
                    torch.ones(imgs.shape[0], latent_dim)).to(device)
                fakebatch = generator(zbatch)
                mixedbatch = torch.cat((imgs, fakebatch))
                indices = torch.randperm(mixedbatch.shape[0]).to(device)

                # set label "1" for fakes for easier processing
                labels = (indices >= imgs.shape[0]).long()

                answer = discriminator(mixedbatch[indices, :])
                neg = torch.ones_like(answer)
                neg[labels>0] = -1
                answer = answer*neg

                loss = -1/imgs.shape[0]*torch.log(labels.float()+answer.T+EPS).sum()
                loss.backward()
                optimizer_D.step()
                
                d_loss += loss.cpu().item()
                batches_done += 1

            for g in range(g_steps):
                optimizer_G.zero_grad()

                zbatch = torch.normal(torch.zeros(imgs.shape[0], latent_dim),
                    torch.ones(imgs.shape[0], latent_dim)).to(device)
                fakebatch = generator(zbatch)
                answer = discriminator(fakebatch)
                loss = -torch.log(answer+EPS).mean()
                loss.backward()
                optimizer_G.step()

                g_loss += loss.cpu().item()

            examples_done = (epoch * (no_iters-1) + batches_done)*bsize

            if ((args.save_interval is not None) and 
                    (examples_done % args.save_interval <= bsize)):
                gen_imgs(generator, latent_dim, device, 25, 
                        'Sample generated image, epoch {}: batch {}'.format(epoch, batches_done+1), 
                        '{}/epoch_{}_batch_{}.png'.format(img_dir, epoch, batches_done+1))

        if args.save_interval is None:
            gen_imgs(generator, latent_dim, device, 25, 
                    'Sample generated image, epoch {}'.format(epoch), 
                    '{}/epoch_{}.png'.format(img_dir, epoch))

        if not (scheduler_G is None or scheduler_D is None):
            scheduler_G.step()
            scheduler_D.step()

        g_loss /= (i+1)*g_steps
        d_loss /= (i+1)*d_steps

        print("[Epoch {}] D-loss: {}, G-loss: {}, LR: {}".format( epoch, d_loss,
            g_loss, optimizer_D.state_dict()['param_groups'][0]['lr']))


def main():
    # Create output image directory
    os.makedirs(args.outpath, exist_ok=True)

    # load data
    if args.t:
        bsize = 2
    else:
        bsize = args.batch_size

    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(args.dpath, train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=bsize, shuffle=True, num_workers=args.num_workers)

    # Initialize device
    device = torch.device(args.device)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim).to(device)
    discriminator = Discriminator(args.latent_dim).to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G,
            args.n_epochs+20)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, 
            args.n_epochs+20)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device,
            args.outpath, args.t, args.dsteps, args.gsteps, args.latent_dim,
            scheduler_G, scheduler_D, bsize)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    if args.sm:
        os.makedirs('saved_models', exist_ok=True)
        import time
        torch.save(generator, "saved_models/gan_generator_{}.pt".format(
            time.time()))
        torch.save(discriminator, "saved_models/gan_discriminator_{}.pt".format(
            time.time()))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=None,
                        help=('save every save_interval training examples.'
                        +'Will be rounded to the batch size.'
                        +'If left empty save every epoch'))
    parser.add_argument('--device', type=str, default="cuda:0", 
            help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('-t', action='store_true',
                        help='run in test mode')
    parser.add_argument('-sm', action='store_true',
                        help='store model data')
    parser.add_argument('--dsteps', type=int, default=1,
                        help='Steps/training iteration for discriminator')
    parser.add_argument('--gsteps', type=int, default=1,
                        help='Steps/training iteration for generator')
    parser.add_argument('--dpath', type=str, default='./data/mnist',
                        help='Root path for dataset')
    parser.add_argument('--outpath', type=str, default='gan_images',
                        help='Path for output images')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers for dataloader')
    args = parser.parse_args()

    if args.t:
        args.save_interval = 15

    main()
