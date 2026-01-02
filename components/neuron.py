import torch
import torch.nn as nn
import torch.nn.functional as F

class LIFNeuron(nn.Module):
    def __init__(self, tau_m=20.0, v_th=1.0, v_rest=0.0, alpha=0.95, device='cpu'):
        super().__init__()
        self.tau_m = tau_m
        self.v_th = v_th
        self.v_rest = v_rest
        self.alpha = alpha  # 膜电位衰减系数
        self.device = device
        
        # 状态变量
        self.v = None      # 膜电位 [batch_size, n_neurons_out]
        self.s = None      # 脉冲输出 [batch_size, n_neurons_out]
        self.p = None      # 资格痕迹 [batch_size, n_neurons_in]
        self.xi = None     # 时序痕迹 [batch_size, n_neurons_in]
    
    def reset(self, batch_size, n_neurons_out):
        """重置神经元状态
        n_neurons_out: 输出神经元数量
        """
        self.v = torch.full((batch_size, n_neurons_out), self.v_rest, device=self.device)
        self.s = torch.zeros((batch_size, n_neurons_out), device=self.device)
        # p和xi的维度会在forward中根据输入调整
        self.p = None
        self.xi = None
    
    def forward(self, x, w, gamma=0.9, sigma=0.1):
        """前向传播
        x: 输入脉冲 [batch_size, n_neurons_in]
        w: 突触权重 [n_neurons_in, n_neurons_out]
        """
        # 确保x在与w相同的设备上
        x = x.to(w.device)
        
        batch_size, n_neurons_in = x.shape
        n_neurons_out = w.shape[1]
        
        # 确保膜电位和脉冲的维度正确
        if self.v is None or self.v.shape[1] != n_neurons_out:
            self.reset(batch_size, n_neurons_out)
        
        # 确保资格痕迹和时序痕迹的维度正确
        if self.p is None or self.p.shape[1] != n_neurons_in:
            self.p = torch.zeros(batch_size, n_neurons_in, device=self.device)
            self.xi = torch.zeros(batch_size, n_neurons_in, device=self.device)
        
        # 更新膜电位
        # v_t = alpha * v_{t-1} - v_th * s_{t-1} + x_t @ w
        v_new = self.alpha * self.v - self.v_th * self.s + x @ w
        
        # 使用可微分的近似函数替代阶跃函数
        # 使用sigmoid函数近似：s_new = 1 / (1 + exp(-k*(v_new - v_th)))
        k = 10  # 陡峭度参数
        s_new = torch.sigmoid(k * (v_new - self.v_th))
        
        # 膜电位重置（软复位）
        v_new = v_new - self.v_th * s_new
        
        # 更新资格痕迹: p_t = alpha * p_{t-1} + x_t
        p_new = self.alpha * self.p + x
        
        # 更新时序痕迹: xi_t = gamma * xi_{t-1} + (1 - gamma) * p_t
        xi_new = gamma * self.xi + (1 - gamma) * p_new
        
        # 计算一致性评分: Gamma_t = exp(-0.5 * ||xi_t - p_t||^2 / sigma^2)
        gamma_t = torch.exp(-0.5 * torch.norm(xi_new - p_new, dim=1) ** 2 / sigma ** 2)
        
        # 更新状态
        self.v = v_new
        self.s = s_new
        self.p = p_new
        self.xi = xi_new
        
        return s_new, gamma_t
