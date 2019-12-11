import argparse
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision import datasets

from a3_gan_template import Generator

NORMAL = False

def sample(config):
    device = torch.device(config.device)
    model = torch.load(config.model_file, map_location=device)
    model.eval()
    while True:
        with torch.no_grad():
            no_images = 2
            zbatch = torch.normal(torch.zeros(no_images, 100), 
                    torch.ones(no_images, 100)).to(device)
            fakebatch = model(zbatch)
            fakebatch = fakebatch/fakebatch.max()
            grid = make_grid(
                fakebatch.cpu().view(no_images, 1, 28, -1).permute(0, 1, 3, 2), 
                nrow = 2, normalize=NORMAL)
            plt.imshow(grid.permute(2, 1, 0).numpy())
            plt.show()
            print("Were images of different class? y/n any other key for exit")
            ans = input()
            if ans == 'y':
                break
            elif ans == 'n':
                pass
            else:
                sys.exit()
    p1 = zbatch[0, :].numpy()
    p2 = zbatch[1, :].numpy()

    nosample = 15
    samples = np.linspace(p1, p2, nosample)
    with torch.no_grad():
        tsamples = torch.from_numpy(samples).view(nosample, -1)
        breakpoint()
        fakebatch = model(tsamples)
    grid = make_grid(
        fakebatch.cpu().view(nosample, 1, 28, -1).permute(0, 1, 3, 2), 
        nrow = 1, normalize=NORMAL)
    plt.imshow(grid.permute(2, 1, 0).numpy())
    plt.show()

            
            


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_file', type=str, required=True, help="Path to an .out file with a saved, trained model")
    parser.add_argument('--device', type=str, default="cpu", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    sample(config)
