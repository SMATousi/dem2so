import numpy as np

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

        iou_list.append(iou)
        dice_list.append(dice)

    # Compute mean IoU and Dice, ignoring NaN values
    mean_iou = np.nanmean(iou_list)
    mean_dice = np.nanmean(dice_list)

    return accuracy, mean_iou, mean_dice