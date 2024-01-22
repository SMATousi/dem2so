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


dem_dir = '/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/DEM2SO/dem2so/dem'
so_dir = '/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/DEM2SO/dem2so/so'

batch_size = arg_batch_size
learning_rate = 0.0001
epochs = arg_epochs
number_of_workers = 1
image_size = arg_imagesize
val_percent = 0.1


transform = RasterTransform()

dataset = RasterTilesDataset(dem_dir=dem_dir, so_dir=so_dir, transform=transform)

# DataLoader

n_val = int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=arg_batch_size, shuffle=True, num_workers=number_of_workers, pin_memory=True)
val_loader = DataLoader(val, batch_size=arg_batch_size, shuffle=False, num_workers=number_of_workers, pin_memory=True, drop_last=True)

print("Data is loaded")



# Instantiate the model
if arg_modelname == 'Unet_1':
    model = UNet_1(n_channels=1, n_classes=9, dropout_rate=0.5).to(device)  # Change n_classes based on your output
if arg_modelname == 'Uformer':
    model = Uformer(img_size=image_size,embed_dim=32,win_size=8,in_chans=1,dd_in=1,token_projection='linear',token_mlp='leff',modulator=False).to(device)
if arg_modelname == 'DepthNet':
    model = DepthNet().to(device)
if arg_modelname == 'Bothnet':
    model = BothNet(in_channels=1, out_channels=1).to(device)  
if arg_modelname == 'Att_Unet':
    model = Att_Unet(n_channels=1, n_classes=1, dropout_rate=0.5).to(device) 
if arg_modelname == 'SA_Unet':
    model = SA_UNet(in_channels=1, num_classes=1).to(device)  


criterion = nn.CrossEntropyLoss()  # Replace with your loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



for epoch in range(epochs):

    train_metrics = {'Train/iou': 0}

    model.train()  # Set the model to training mode
    for batch in tqdm(train_loader):
        dem_image = batch['DEM']
        so_image = batch['SO']
        # print(dem_image.shape)
        # print(so_image.shape)
        dem_image, so_image = dem_image.to(device), so_image.to(device)  # Move data to GPU
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(dem_image)  # Forward pass
        loss = criterion(outputs, so_image)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        iou = mIOU(so_image, outputs)
        # train_metrics['Train/accuracy'] += acc
        train_metrics['Train/iou'] += iou
        # train_metrics['Train/dice'] += dice

        if arg_nottest:
            continue
        else:
            break

    if arg_nottest:
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
    
    if args.logging:
        wandb.log(train_metrics)
        wandb.log({'Train/Loss':loss.item()})
    
        
    
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item()}")
    print(train_metrics)

    

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_metrics = {'Validation/iou': 0}
    with torch.no_grad():
        val_correct = 0
        val_total = 0
        for batch in tqdm(val_loader):
            dem_image = batch['DEM']
            so_image = batch['SO']
            dem_image, so_image = dem_image.to(device), so_image.to(device)  # Move data to GPU
            val_outputs = model(dem_image)  # Forward pass
            loss = criterion(val_outputs, so_image)
            iou = mIOU(so_image, val_outputs)
            # val_metrics['Validation/accuracy'] += acc
            val_metrics['Validation/iou'] += iou
            # val_metrics['Validation/dice'] += dice

            if arg_nottest:
                continue
            else:
                break
        
        if arg_nottest:
            for k in val_metrics:
                val_metrics[k] /= len(val_loader)

        if args.logging:
            wandb.log(val_metrics)
            wandb.log({'Validation/Loss':loss.item()})

            if (epoch + 1) % arg_savingstep == 0:

                torch.save(model.state_dict(), f'./model_epoch_{epoch+1}.pth')
                artifact = wandb.Artifact(f'model_epoch_{epoch+1}', type='model')
                artifact.add_file(f'./model_epoch_{epoch+1}.pth')
                wandb.log_artifact(artifact)
                # save_comparison_figures(model, val_loader, epoch + 1, device)

        print(val_metrics)