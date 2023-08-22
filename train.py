import os 
import torch 
import argparse 
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets 
from torchvision.utils import  make_grid 
import numpy as np 
import random 
import multiprocessing as mp
from matplotlib import pyplot as plt 
from tqdm import tqdm 
 
from model import VQModel 


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(data_loader, model, optimizer, args, data_variance=1):
    """trianing the model""" 
    model.train() 
    iteration = 0 
    loss_cum = 0
    with tqdm(enumerate(data_loader), total=len(data_loader)) as t:
        for _, batch in t:
            images, _ = batch
            images = images.to(args.device)
            optimizer.zero_grad()
            x, loss_vq, perplexity, _ = model(images)

            # loss function
            loss_recons = F.mse_loss(x, images) / data_variance
            loss = loss_recons + loss_vq
            loss.backward()

            optimizer.step()
            loss_cum += loss.item() 
            iteration += 1

            t.set_postfix(loss=loss_cum / iteration)
            args.steps +=1 
            if args.debug == True:
                break 


def test(data_loader, model, args, ):
    """evaluation model"""
    model.eval()
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for images, _ in data_loader:
            images = images.to(args.device)
            x, loss, _, _ = model(images)
            loss_recons += F.mse_loss(x, images)
            loss_vq += loss 
            if args.debug == True: 
                break 

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    return loss_recons.item(), loss_vq.item()


def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.to(args.device)
        x, _, _, _ = model(images)
    return x


def image_plot(image_tensor, epoch, idx): 
    image_np = image_tensor.numpy() 
    plt.imshow(np.transpose(image_np, (1, 2, 0)), interpolation="nearest") 
    # plt.show() 
    image_path = str(epoch) + '_' + str(idx) + '.png' 
    image_path = os.path.join('result/sample', image_path) 
    plt.savefig(image_path) 


def main(args): 
    seed_everything(args.seed) 
    save_filename = 'ckpt'

    # load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    train_dataset = datasets.CIFAR10(args.data_folder,
        train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(args.data_folder,
        train=False, download=False, transform=transform)
    valid_dataset = test_dataset 

    data_variance=np.var(train_dataset.data / 255.0)
    num_channels = 3

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, 
        worker_init_fn=seed_worker, generator=g) 
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=32, shuffle=False,
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
        loss_rec, loss_vq = test(valid_loader, model, args,) 
        print('reconstruct loss: ', loss_rec)
        print('quantise loss', loss_vq) 

        # visualization
        images, _ = next(iter(test_loader))
        rec_images = generate_samples(images, model, args)
        input_grid = make_grid(images, nrow=8, range=(-1, 1), normalize=True)
        rec_grid = make_grid(rec_images, nrow=8, range=(-1,1), normalize=True) 

        image_plot(input_grid, epoch, 0) 
        image_plot(rec_grid, epoch, 1)
        # save model
        if (epoch == 0) or (loss_rec < best_loss):
            best_loss = loss_rec
            with open('{0}/best.pt'.format(save_filename), 'wb') as f:
                torch.save(model.state_dict(), f) 
        if args.debug == True: 
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
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    
    args = parser.parse_args()
    # Device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.steps = 0 
    args.debug = True 
    main(args)
