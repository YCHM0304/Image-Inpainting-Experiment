import torch
import torch.nn as nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def laplacian_gradient(self, x):
        """計算拉普拉斯梯度
        Args:
            x: 輸入張量 [B, C, H, W]
        Returns:
            梯度張量
        """
        # 定義拉普拉斯卷積核
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        if x.is_cuda:
            laplacian_kernel = laplacian_kernel.cuda()

        # 對每個通道進行卷積
        channels = []
        for channel in range(x.shape[1]):
            pad = F.pad(x[:, channel:channel+1], (1, 1, 1, 1), mode='reflect')
            grad = F.conv2d(pad, laplacian_kernel)
            channels.append(grad)

        return torch.cat(channels, dim=1)

    def forward(self, I_pred, I_s, I_gt, mask):
        """計算邊界 loss
        Args:
            I_pred: 預測的無陰影圖像
            I_s: 有陰影的圖像
            I_gt: 真實的無陰影圖像
            mask: 陰影區域遮罩 (1表示陰影區域)
        """
        # 計算梯度
        grad_pred = self.laplacian_gradient(I_pred)
        grad_s = self.laplacian_gradient(I_s)
        grad_gt = self.laplacian_gradient(I_gt)

        # 計算非陰影區域的梯度損失
        non_stained_loss = F.mse_loss(grad_pred, grad_s, reduction='none') * (1 - mask)

        # 計算陰影區域的梯度損失
        stained_loss = F.mse_loss(grad_pred, grad_gt, reduction='none') * mask

        # 合併損失
        total_loss = non_stained_loss + stained_loss

        # 返回平均損失
        return total_loss.mean()

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)