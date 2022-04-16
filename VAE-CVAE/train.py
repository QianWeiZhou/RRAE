import os
import time
import torch
import argparse

import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import VAE




def get_inputx(x_input_list, multi, c=None):
    X_input = x_input_list[0]
    for X_input_i in x_input_list[1:]:
        X_input = torch.cat((X_input, X_input_i),1)

    for x in range(multi - len(x_input_list)):
        X_input = torch.cat((X_input, torch.zeros_like(x_input_list[0])),1)
    return X_input

def vae_encode_decode(x, y, vae, multi, conditional):
    c = y if conditional else None

    x_input_list = [x]

    for i in range(multi - 1):  
        x_input = get_inputx(x_input_list, multi, c)
        recon_x, _, _, _ = vae(x_input, c)
        x_input_list.append((x - recon_x)/2)


    x_input = get_inputx(x_input_list, multi, c)
    recon_x, mean, log_var, z = vae(x_input, c)
    return recon_x, mean, log_var, z

def eval_vae(test_loader, vae, conditional, multi, device):
    log_loss_a = 0

    for iteration, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        recon_x, _, _, _ = vae_encode_decode(x, y, vae, multi=multi, conditional=conditional)

        log_loss_a += F.mse_loss(recon_x, x)

    return (log_loss_a/len(test_loader)).item()


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()
    list_test_loss=[]

    dataset = MNIST(
        root='/home/work/tp/NearLoss/data/MNIST/', train=True, transform=transforms.ToTensor(),
        download=True)
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = MNIST(
        root='/home/work/tp/NearLoss/data/MNIST/', train=False, transform=transforms.ToTensor(),
        download=True)
    test_loader = DataLoader(
        dataset=testset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(
            recon_x.view(-1, 28*28), x.view(-1, 28*28), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=10 if args.conditional else 0,
        multi=args.multi).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            recon_x, mean, log_var, z = vae_encode_decode(x, y, vae, multi=args.multi, conditional=args.conditional)
            
            loss = loss_fn(recon_x, x, mean, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration == len(data_loader)-1:
                test_loss = eval_vae(test_loader, vae, args.conditional, args.multi, device)
                list_test_loss.append(test_loss)
                print("Epoch {:03d}/{:03d}, Loss {:9.5f}".format(epoch, args.epochs, test_loss))

       
    list_test_loss = np.array(list_test_loss)
    print(f'min epoch: {list_test_loss.argmin()}')
    print(f'min loss: {list_test_loss.min()}')

    file = f'{args.checkfolder}/log.txt'
    with open(file, 'a+') as f:
      f.write('conditional:{}, z-{:02d}_m{}, min loss: {:.6f}, min epoch: {:03d}/{:03d}\n'.format(
            str(args.conditional).ljust(5, ' '), args.latent_size, args.multi, float(list_test_loss.min()), list_test_loss.argmin()+1, args.epochs))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=1e10)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true', default=False)
    parser.add_argument('-m', '--multi', type=int, default=1)
    parser.add_argument("--checkfolder", type=str, default='./')
    parser.add_argument('-gpu', '--gpu', default="0", help="gpu number")

    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    main(args)
