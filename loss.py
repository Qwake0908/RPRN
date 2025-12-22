import torch
import torch.nn as nn

class RDPLLoss(nn.Module):
    """奖励驱动的预测编码时序痕迹损失函数（RDPL）"""
    def __init__(self, mu=0.1):
        """初始化损失函数
        mu: 正则系数
        """
        super().__init__()
        self.mu = mu
    
    def forward(self, x, x_hat, p, gamma, R):
        """计算损失
        x: 实际输入脉冲 [batch_size, time_steps, n_neurons]
        x_hat: 预测输入脉冲 [batch_size, time_steps, n_neurons]
        p: 资格痕迹 [batch_size, time_steps, n_neurons]
        gamma: 一致性评分 [batch_size, time_steps]
        R: 奖励信号 [batch_size, time_steps]
        """
        # 确保所有张量都有梯度
        x = x.clone().requires_grad_(True)
        x_hat = x_hat.clone().requires_grad_(True)
        p = p.clone().requires_grad_(True)
        gamma = gamma.clone().requires_grad_(True)
        R = R.clone().requires_grad_(True)
        
        batch_size, time_steps, n_neurons = x.shape
        
        # 计算预测误差 ||x - x_hat||^2
        pred_error = torch.sum((x - x_hat) ** 2, dim=2)
        
        # 计算匹配误差项 (1 - gamma)||p||^2
        match_error = self.mu * (1 - gamma) * torch.sum(p ** 2, dim=2)
        
        # 总损失: 1/2 * sum_t R_t (pred_error + match_error)
        loss = 0.5 * torch.sum(R * (pred_error + match_error)) / (batch_size * time_steps)
        
        return loss
