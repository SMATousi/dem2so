import numpy as np
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
from torch.nn import functional as F

def calculate_metrics(predicted, desired, num_classes=9):
    """
    Calculate accuracy, mean IoU and mean Dice coefficient for one-hot encoded predicted map 
    and single-channel desired map.

    :param predicted: Predicted tensor (one-hot encoded).
    :param desired: Desired (ground truth) tensor (single-channel with class labels).
    :param num_classes: Number of classes in the data.
    :return: Accuracy, mean IoU, mean Dice coefficient.
    """
    predicted = predicted.cpu().detach().numpy()
    desired = desired.cpu().detach().numpy()

    # Convert desired map to one-hot encoding
    desired_one_hot = np.eye(num_classes)[desired.astype(np.int32)]
    desired_one_hot = desired_one_hot.transpose(0,3,2,1)
    # print(desired_one_hot.shape)
    # print(predicted.shape)

    accuracy = np.mean(np.argmax(predicted, axis=1) == np.argmax(desired_one_hot, axis=1))

    iou_list = []
    dice_list = []

    for cls in range(num_classes):
        predicted_cls = predicted[:, cls, :, :]
        desired_cls = desired_one_hot[:, cls, :, :]

        intersection = np.logical_and(predicted_cls, desired_cls)
        union = np.logical_or(predicted_cls, desired_cls)

        if np.sum(union) == 0:  # Avoid division by zero
            iou = np.nan
            dice = np.nan
        else:
            iou = np.sum(intersection) / np.sum(union)
            dice = 2 * np.sum(intersection) / (np.sum(predicted_cls) + np.sum(desired_cls))

        print(iou)
        iou_list.append(iou)
        dice_list.append(dice)

    # Compute mean IoU and Dice, ignoring NaN values
    mean_iou = np.nanmean(iou_list)
    mean_dice = np.nanmean(dice_list)

    return accuracy, mean_iou, mean_dice


def mIOU(label, pred, num_classes=9):
    pred = F.softmax(pred, dim=1)              
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
            # print(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)