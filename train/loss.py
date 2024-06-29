import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.nn.modules.loss import _Loss


def MSE(para):
    """
    L2 loss
    """
    return nn.MSELoss()


def L1(para):
    """
    L1 loss
    """
    return nn.L1Loss()


class L1GradientLoss(_Loss):
    """
    Gradient loss
    """

    def __init__(self):
        super(L1GradientLoss, self).__init__()
        self.get_grad = Gradient()
        self.L1 = nn.L1Loss()

    def forward(self, x, y):
        grad_x = self.get_grad(x)
        grad_y = self.get_grad(y)
        loss = self.L1(grad_x, grad_y)
        return loss

class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        #x1 = x[:, 1]
        #x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        #x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        #x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        #x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        #x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        #x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        #x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        #x = torch.cat([x0, x1, x2], dim=1)
        x = x0
        return x
    
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        計算二元交叉熵損失
        :param predictions: 預測值，取值在0到1之間
        :param targets: 真實標籤，取值為0或1
        :return: 二元交叉熵損失
        """
        # print(predictions.shape, targets.shape)
        predictions = torch.clamp(predictions, min=1e-7, max=1-1e-7)  # 避免log(0)的情況
        loss = -targets * torch.log(predictions) - (1 - targets) * torch.log(1 - predictions)
        # print(loss.mean())
        return loss.mean()  # 返回平均損失
    
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        weight_for_class_0 = 0.2
        weight_for_class_1 = 0.8
        self.weights = torch.tensor([weight_for_class_0, weight_for_class_1])

    def forward(self, inputs, targets):
        inputs_sq = torch.squeeze(inputs)
        return F.cross_entropy(inputs_sq, targets, weight=self.weights)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        return 1 - dice_score
class CombinedFocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5):
        super(CombinedFocalDiceLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        inputs_sq = torch.squeeze(inputs)
        dice_loss = self.dice_loss(inputs_sq, targets)
        focal_loss = self.focal_loss(inputs_sq, targets)
        # print(self.dice_weight * dice_loss + self.focal_weight * focal_loss)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss
    
class CombinedAllLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.25, focal_weight=0.25, bce_weight = 0.5):
        super(CombinedAllLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.bce_loss = BinaryCrossEntropyLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        # inputs_sq = torch.squeeze(inputs)
        dice_loss = self.dice_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        bce_loss = self.bce_loss(inputs, targets)
        # print(self.dice_weight * dice_loss + self.focal_weight * focal_loss)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss +self.bce_weight*bce_loss
    
class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        print("You are using `CEL`!")
        self.eps = 1e-6

    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)