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



class GradientLoss(nn.Module):
    def __init__(self, weight_gradient=0.5, tolerance=0.00, weight_pixel=1.0):
        super(GradientLoss, self).__init__()
        # Weight of the gradient loss component
        self.weight_gradient = weight_gradient
        # Tolerance for comparing gradient magnitudes
        self.tolerance = tolerance
        # Weight of the pixel-wise loss component (e.g., L1 loss for image-to-image comparison)
        self.weight_pixel = weight_pixel
        self.pixel_loss = nn.CrossEntropyLoss()  # L1 loss for pixel-wise comparison

    def forward(self, predictions, labels):
        """
        Calculate the custom loss for image prediction tasks, focusing on pixel-wise accuracy and gradient similarity.

        :param predictions: The predicted images.
        :param labels: The target images.
        :return: The combined loss value, along with individual loss components for monitoring.
        """
        # Compute the pixel-wise loss between predictions and labels
        pixel_loss = self.pixel_loss(predictions, labels)

        # Compute the gradient magnitude for both predictions and labels
        pred = torch.softmax(predictions, dim=1)
        pred = torch.argmax(pred, dim=1)

        labels_grad_mag = calculate_gradient_magnitude(labels)
        predictions_grad_mag = calculate_gradient_magnitude(pred)

        # Compute the absolute difference between the two gradient magnitudes
        diff = torch.square(labels_grad_mag - predictions_grad_mag)

        # Compute the mean of the differences as the gradient loss
        gradient_loss = diff.mean()

        # Combine the losses with their respective weights
        combined_loss = self.weight_pixel * pixel_loss + self.weight_gradient * gradient_loss

        return combined_loss, pixel_loss, gradient_loss


def calculate_gradient_magnitude(image):
    image = torch.unsqueeze(image, 1)
    """Calculate the gradient magnitude of an image using the Sobel operator."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
    
    grad_x = F.conv2d(image.type(torch.float32), sobel_x, padding=1)
    grad_y = F.conv2d(image.type(torch.float32), sobel_y, padding=1)
    
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    return gradient_magnitude

def compare_gradients(so, predictions, tolerance=0.00):
    """Compare the gradient magnitudes of SO and predictions, returning the percentage of matches."""
    so_grad_mag = calculate_gradient_magnitude(so)
    predictions_grad_mag = calculate_gradient_magnitude(pred)
    
    # Compute the absolute difference between the two gradient magnitudes
    diff = torch.abs(so_grad_mag - predictions_grad_mag)
    
    # Determine matches based on the tolerance threshold
    matches = diff <= tolerance
    
    # Calculate the percentage of matches
    match_percentage = matches.float().mean().item() * 100
    
    return match_percentage, diff

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