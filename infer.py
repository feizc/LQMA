import os 
import argparse 
import numpy as np 

import torch 
from torchvision import transforms, datasets 
import imageio

from model import VQModel 



def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy.reshape(image_numpy.shape[0], image_numpy.shape[1])

    imageio.imwrite(image_path, image_numpy)



def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        if image_tensor.dim() == 4:
            image_numpy = ((image_tensor[0]+1.0)/2.0).clamp(0,1).cpu().float().numpy()
        else:
            image_numpy = ((image_tensor+1.0)/2.0).clamp(0,1).cpu().float().numpy() # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)



if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='LMAE')
    # General
    parser.add_argument('--data_folder', type=str, default='data/cifar', help='name of the data folder')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size (default: 16)')
    # Latent space
    parser.add_argument('--hidden_size', type=int, default=128, help='size of the latent vectors (default: 128)')
    parser.add_argument('--num_residual_hidden', type=int, default=32, help='size of the redisual layers (default: 32)')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='number of residual layers (default: 2)')
    # Quantiser parameters
    parser.add_argument('--embedding_dim', type=int, default=64, help='dimention of codebook (default: 64)')
    parser.add_argument('--num_embedding', type=int, default=512, help='number of codebook (default: 512)')
    parser.add_argument('--distance', type=str, default='l2', help='distance for codevectors and features')
    # Miscellaneous
    parser.add_argument('--num_workers', type=int, default=1, help='number of workers for trajectories sampling (default: 1）')
    parser.add_argument('--device', type=str, default= torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='set the device (cpu or cuda, default: cpu)')
    args = parser.parse_args() 

    # load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(args.data_folder, train=False, download=False, transform=transform)
    num_channels = 3 

    # Define the dataloaders
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) 
    # Define the model
    model = VQModel(num_channels, args.hidden_size, args.num_residual_layers, args.num_residual_hidden,
                    args.num_embedding, args.embedding_dim, distance=args.distance) 
    # load model
    ckpt = torch.load('./ckpt/best.pt') 
    model.load_state_dict(ckpt) 

    model = model.to(args.device) 
    model.eval() 

    # store results 
    results_path = 'result'
    original_path = os.path.join(results_path, 'original')
    rec_path = os.path.join(results_path, 'rec')
    if not os.path.exists(original_path):
        os.makedirs(original_path)
        os.makedirs(rec_path)
    
    # test model
    encodings = []
    indexes = []
    labels = []
    all_images = []
    imageid = 0
    for images, label in test_loader:
        images = images.to(args.device)
        x_recons, loss, perplexity, encoding = model(images)
        # save indexes
        index = encoding.argmax(dim=1).view(images.size(0), -1)
        indexes.append(index)
        all_images.append(images.view(images.size(0), -1))
        # save labels
        labels.append(label)
        # save encodings
        encodings.append(encoding)
        # save image
        for x_recon, image in zip(x_recons, images):
            x_recon = tensor2im(x_recon)
            image = tensor2im(image)
            name = str(imageid).zfill(8) + '.jpg'
            save_image(image, os.path.join(original_path, name))
            save_image(x_recon, os.path.join(rec_path, name))
            imageid += 1
        break 
    



    


