import torch 
import argparse 
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np 
import random 
import multiprocessing as mp

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from model import VQModel 

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(data_loader, model, optimizer, args, data_variance=1):
    """trianing the model"""
    for images, _ in data_loader:
        images = images.to(args.device)
        optimizer.zero_grad()
        x, loss_vq, perplexity, _ = model(images)

        # loss function
        loss_recons = F.mse_loss(x, images) / data_variance
        loss = loss_recons + loss_vq
        loss.backward()

        optimizer.step()
        print(loss)
        args.steps +=1
        break 


def main(args): 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    train_dataset = datasets.CIFAR10(args.data_folder,
        train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(args.data_folder,
        train=False, download=False, transform=transform)
    data_variance=np.var(train_dataset.data / 255.0)
    num_channels = 3

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, 
        worker_init_fn=seed_worker, generator=g)
    
    # Define the model
    model = VQModel(num_channels, args.hidden_size, args.num_residual_layers, args.num_residual_hidden,
        args.num_embedding, args.embedding_dim, args.commitment_cost, args.distance,)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 

    # Update the model
    best_loss = -1.
    for epoch in range(args.num_epochs):
        # training and testing the model
        train(train_loader, model, optimizer, args, data_variance)  
        break 




if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='LQAE') 
    # General 
    parser.add_argument('--data_folder', type=str, default='data/cifar', help='name of the data folder')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 1024)')
    # Latent space
    parser.add_argument('--hidden_size', type=int, default=128, help='size of the latent vectors (default: 128)')
    parser.add_argument('--num_residual_hidden', type=int, default=32, help='size of the redisual layers (default: 32)')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='number of residual layers (default: 2)')
    # Quantiser parameters
    parser.add_argument('--embedding_dim', type=int, default=64, help='dimention of codebook (default: 64)')
    parser.add_argument('--num_embedding', type=int, default=512, help='number of codebook (default: 512)')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='hyperparameter for the commitment loss')
    parser.add_argument('--distance', type=str, default='l2', help='distance for codevectors and features')
    # Optimization
    parser.add_argument('--seed', type=int, default=42, help="seed for everything")
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count() - 1, help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    
    args = parser.parse_args()
    # Device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.steps = 0 
    main(args)