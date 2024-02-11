import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
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

def setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)



parser = argparse.ArgumentParser(description="A script with argparse options")

# Add an argument for an integer option
parser.add_argument("--runname", type=str, required=False)
parser.add_argument("--projectname", type=str, required=False)
parser.add_argument("--modelname", type=str, required=False)
parser.add_argument("--batchsize", type=int, default=4)
parser.add_argument("--savingstep", type=int, default=100)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--imagesize", type=int, default=256)
parser.add_argument("--threshold", type=float, default=1)
parser.add_argument("--dropoutrate", type=float, default=0.5)
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
arg_dropoutrate = args.dropoutrate

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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print(device)

dem_dir = '/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/DEM2SO/New_Data/dem'
so_dir = '/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/DEM2SO/New_Data/so'
rgb_dir = '/home/macula/SMATousi/Gullies/ground_truth/google_api/training_process/DEM2SO/New_Data/rgb'

#dem_dir = '/root/home/data/dem'
#so_dir = '/root/home/data/so'
#rgb_dir = '/root/home/data/rgb'

# pretrained_model_path = '/root/home/pre_trained/B3_rn50_moco_0099_ckpt.pth'
pretrained_model_path = '/home/macula/SMATousi/cluster/docker-images/dem2so_more_data/pre_models/B3_rn50_moco_0099_ckpt.pth'


batch_size = arg_batch_size
learning_rate = 0.0001
epochs = arg_epochs
number_of_workers = 1
image_size = arg_imagesize
val_percent = 0.1


transform = RGB_RasterTransform()

dataset = RGB_RasterTilesDataset(dem_dir=dem_dir, so_dir=so_dir, rgb_dir=rgb_dir, transform=transform)

def prepare(input_dataset, rank, world_size, batch_size=arg_batch_size, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(input_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader = DataLoader(input_dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

# DataLoader

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):

    setup(rank, world_size)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])


    train_loader = prepare(train, rank, world_size)
    val_loader = prepare(val, rank, world_size)

    print("Data is loaded")


    model = RGB_DEM_to_SO(resnet_output_size=(8, 8), 
                            fusion_output_size=(128, 128), 
                            model_choice = arg_modelname, 
                            resnet_saved_model_path=pretrained_model_path,
                            dropout_rate=arg_dropoutrate).to(rank)
    
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    from torch.optim import Adam
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop

    for epoch in range(epochs):

        train_metrics = {'Train/iou': 0}
        
        for i, batch in enumerate(tqdm(train_loader)):
            dem = batch['DEM'].to(device)
            so = batch['SO'].to(device)
            rgbs = [batch['RGB'][k].to(device) for k in range(6)]

            # Forward pass
            outputs = model(dem, rgbs)
            loss = criterion(outputs, so)
            iou = mIOU(so, outputs)
            train_metrics['Train/iou'] += iou

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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

            for i, batch in enumerate(tqdm(val_loader)):
                dem = batch['DEM'].to(device)
                so = batch['SO'].to(device)
                rgbs = [batch['RGB'][k].to(device) for k in range(6)]

                outputs = model(dem, rgbs)
                loss = criterion(outputs, so)
                iou = mIOU(so, outputs)
                val_metrics['Validation/iou'] += iou


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
                    
                    os.makedirs('../saved_models', exist_ok=True)
                    torch.save(model.state_dict(), f'../saved_models/model_epoch_{epoch+1}.pth')
                    artifact = wandb.Artifact(f'model_epoch_{epoch+1}', type='model')
                    artifact.add_file(f'../saved_models/model_epoch_{epoch+1}.pth')
                    wandb.log_artifact(artifact)
                    # save_comparison_figures(model, val_loader, epoch + 1, device)

            print(val_metrics)
    
    cleanup()


if __name__ == '__main__':

    world_size = 4        
    mp.spawn(
        main,
        args=(world_size),
        nprocs=world_size
    )

