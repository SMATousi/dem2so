import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.01):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions, targets):
        # Compute the Mean Squared Error (MSE) loss
        mse_loss = self.mse_loss(predictions, targets)
        
        # Compute the penalty term
        penalty = torch.mean(torch.abs(predictions))
        
        # Combine the MSE loss with the penalty term
        loss = mse_loss + self.alpha * penalty
        
        return loss
