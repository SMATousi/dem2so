import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
import random
import numpy as np
from tqdm import tqdm
import argparse
import wandb
from model import *
from dataset import *
from utils import *



random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="A script with argparse options")

# Add an argument for an integer option
parser.add_argument("--runname", type=str, required=False)
parser.add_argument("--projectname", type=str, required=False)
parser.add_argument("--modelname", type=str, required=True)
parser.add_argument("--batchsize", type=int, default=4)
parser.add_argument("--savingstep", type=int, default=100)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--imagesize", type=int, default=256)
parser.add_argument("--threshold", type=float, default=1)
parser.add_argument("--nottest", help="Enable verbose mode", action="store_true")
parser.add_argument("--logging", help="Enable verbose mode", action="store_true")

args = parser.parse_args()

arg_batch_size = args.batchsize
arg_epochs = args.epochs
arg_runname = args.runname
arg_projectname = args.projectname
arg_modelname = args.modelname
arg_savingstep = args.savingstep
arg_threshold = args.threshold
arg_imagesize = args.imagesize

if args.nottest:
    arg_nottest = True 
else:
    arg_nottest = False


args = parser.parse_args()

if args.logging:

    wandb.init(
            # set the wandb project where this run will be logged
        project=arg_projectname, name=arg_runname
            
            # track hyperparameters and run metadata
            # config={
            # "learning_rate": 0.02,
            # "architecture": "CNN",
            # "dataset": "CIFAR-100",
            # "epochs": 20,
            # }
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

dem_dir = '/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/DEM2SO/dem2so/dem_with_rgb/dem'
so_dir = '/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/DEM2SO/dem2so/dem_with_rgb/so'
rgb_dir = '/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/DEM2SO/dem2so/dem_with_rgb/rgb'


batch_size = arg_batch_size
learning_rate = 0.0001
epochs = arg_epochs
number_of_workers = 1
image_size = arg_imagesize
val_percent = 0.1


transform = RasterTransform()

dataset = RGB_RasterTilesDataset(dem_dir=dem_dir, so_dir=so_dir, rgb_dir=rgb_dir, transform=transform)

# DataLoader

n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=arg_batch_size, shuffle=True, num_workers=number_of_workers, pin_memory=True)
val_loader = DataLoader(val, batch_size=arg_batch_size, shuffle=False, num_workers=number_of_workers, pin_memory=True, drop_last=True)

print("Data is loaded")


