import torch
import torch.nn as nn
from scipy.stats import truncnorm
import numpy as np

class NeuronClassNumPy():

    def __init__(self, par):
        """
        Args:
        - par: object containing the parameters of the neuron
        """
        self.par = par  
        self.alpha = (1 - self.par.dt / self.par.tau_m)              
        
        # 初始化二维坐标位置编码相关参数
        # 神经元自身位置 (x, y)
        self.neuron_position = getattr(par, 'neuron_position', (0.0, 0.0))  # 神经元在地图中的位置
        # 目标奖励位置 (x, y)
        self.target_position = getattr(par, 'target_position', (0.0, 0.0))  # 目标奖励位置
        # 奖励参数
        self.reward_strength = getattr(par, 'reward_strength', 1.0)  # 奖励强度
        self.reward_sigma = getattr(par, 'reward_sigma', 1.0)  # 奖励分布的标准差
        
        # 异序列抑制参数
        self.mu = getattr(par, 'mu', 0.0)  # 异序列抑制强度，默认0
        self.gamma = getattr(par, 'gamma', 0.95)  # 时序迹更新系数
        self.sigma_p = getattr(par, 'sigma_p', 0.15)  # 资格迹差异容限

    def initialize(self):
        """
        supported initialization methods:
        - 'trunc_gauss': Initialize the weights using a truncated Gaussian distribution.
        - 'uniform': Initialize the weights using a uniform distribution.
        - 'fixed': Initialize the weights with a fixed value.
        """
        if self.par.init == 'trunc_gauss':
            a = (self.par.init_a - self.par.init_mean) / (1 / np.sqrt(self.par.N))
            b = (self.par.init_b - self.par.init_mean) / (1 / np.sqrt(self.par.N))
            self.w = truncnorm(a, b, loc=self.par.init_mean, scale=1/np.sqrt(self.par.N)).rvs(self.par.N)
        elif self.par.init == 'uniform':
            self.w = np.random.uniform(0, self.par.init_mean, self.par.N)
        elif self.par.init == 'fixed':
            self.w = self.par.init_mean * np.ones(self.par.N)
        
        # 初始化时序迹
        self.xi = np.zeros(self.par.N)  # 时序迹，记录资格迹的统计平均模式
        
    def state(self):
        """
        state variables:
        - self.v: Array of shape (self.par.batch,) representing the voltage.
        - self.z: Array of shape (self.par.batch,) representing the output spikes.
        - self.p: Array of shape (self.par.batch, self.par.N) representing the eligibiliy traces.
        - self.epsilon: Array of shape (self.par.batch, self.par.N) representing the prediction errors.
        - self.E: Array of shape (self.par.batch,) representing the global signal.
        - self.grad: Array of shape (self.par.batch, self.par.N) representing the gradient.
        - self.R: Array of shape (self.par.batch,) representing the reward signal.
        - self.Gamma: Array of shape (self.par.batch,) representing the sequence consistency score.
        """
        self.v = np.zeros(self.par.batch)
        self.z = np.zeros(self.par.batch)
        self.p = np.zeros((self.par.batch, self.par.N))
        self.epsilon = np.zeros((self.par.batch, self.par.N))
        self.E = np.zeros(self.par.batch)
        self.grad = np.zeros((self.par.batch, self.par.N))
        self.R = np.ones(self.par.batch)  # 默认为1，表示无奖励调制
        self.Gamma = np.ones(self.par.batch)  # 默认为1，表示完全匹配
    
    def __call__(self, x):
        """
        Args:
        x : Array of shape (self.par.batch, self.par.N)
        
        Updates:
        - self.v: Array of shape (self.par.batch,) 
        - self.z: Array of shape (self.par.batch,)
        """
        self.v = self.alpha * self.v + np.einsum('ij,j->i', x, self.w) \
                 - self.par.v_th * self.z
        self.z = np.zeros(self.par.batch)
        self.z[self.v - self.par.v_th > 0] = 1
    
    def backward_online(self, x):
        """
        Args:
        x : Array of shape (self.par.batch, self.par.N)
    
        Updates:
        - self.epsilon: Array of shape (self.par.batch, self.par.N) 
        - self.E: Array of shape (self.par.batch,) 
        - self.grad: Array of shape (self.par.batch, self.par.N) 
        - self.p: Array of shape (self.par.batch, self.par.N)
        - self.R: Array of shape (self.par.batch,) - 奖励信号
        - self.Gamma: Array of shape (self.par.batch,) - 序列一致性评分
        - self.xi: Array of shape (self.par.N) - 更新时序迹
        """
        self.epsilon = x - np.einsum('i,j->ij', self.v, self.w)
        self.E = np.einsum('ij,j->i', self.epsilon, self.w)
        
        # 计算基于位置的奖励信号
        self._compute_reward()
        
        # 计算基础梯度（不包含异序列抑制项）
        base_grad = - np.einsum('i,ij->ij', self.v, self.epsilon) \
                    - np.einsum('i,ij->ij', self.E, self.p)
        
        # 先更新资格迹
        self.p = self.alpha * self.p + x
        
        # 更新时序迹：使用资格迹的平均值作为当前时刻的资格迹
        avg_p = self.p.mean(axis=0)  # 计算批次平均的资格迹
        self.xi = self.gamma * self.xi + (1 - self.gamma) * avg_p  # 更新时序迹
        
        # 计算序列一致性评分Gamma
        # 对每个批次样本，计算当前p与xi的相似性
        self.Gamma = np.ones(self.par.batch)  # 默认完全匹配
        if self.mu > 0:  # 只有当mu>0时才计算一致性评分
            # 向量化计算每个批次和突触的资格迹与时序迹的高斯相似度
            similarities = np.exp(-0.5 * ((self.p - self.xi[None, :]) / self.sigma_p) ** 2)
            # 计算平均相似度作为Gamma
            self.Gamma = np.mean(similarities, axis=1)
        
        # 添加异序列抑制项（当mu>0时）
        if self.mu > 0:
            # 异序列抑制项：mu*(1-Gamma)*p
            suppression_term = self.mu * np.einsum('i,ij->ij', (1 - self.Gamma), self.p)
            base_grad += suppression_term
        
        # 使用奖励信号调制梯度
        self.grad = base_grad * self.R[:, None]  # 应用奖励调制
    
    def _compute_reward(self):
        """
        计算基于二维坐标的奖励信号
        神经元放电时，根据其在地图中的绝对位置与目标位置的距离给予奖励
        """
        # 计算二维欧几里得距离
        dx = self.neuron_position[0] - self.target_position[0]
        dy = self.neuron_position[1] - self.target_position[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # 使用高斯函数计算奖励：距离越近，奖励越大
        # 奖励范围在1到1+reward_strength之间
        self.R = 1.0 + self.reward_strength * np.exp(-0.5 * (distance / self.reward_sigma) ** 2)
        
        # 如果神经元发放了，增强奖励效果
        self.R = np.where(self.z > 0, self.R * 1.5, self.R)
    
    def _compute_consistency(self):
        """
        计算当前资格迹模式与时序迹的相似性
        """
        # 计算每个突触的匹配度
        similarities = np.exp(-(self.p - self.xi)**2 / (2 * self.sigma_p**2))
        
        # 归一化匹配度到[0,1]范围
        Gamma = np.mean(similarities, axis=1)
        
        return Gamma
    
    def update_online(self):
        """
        'soft': the weight (w) is updated using a proportional term of 
        the gradient, scaled by the learning rate (eta) and the mean of 
        the gradient along the axis 0.

        'hard': the weight (w) is updated by subtracting the scaled gradient 
        (eta * grad.mean(axis=0)) from the weight (w). Then, any negative values 
        in the updated weight (w) are set to zero, enforcing a hard lowerbound 
        on weight values.

        'else':the weight (w) is updated by subtracting the scaled gradient.
        """
        if self.par.bound == 'soft':
            self.w = self.w - self.w * self.par.eta * self.grad.mean(axis=0)
        elif self.par.bound == 'hard':
            self.w = self.w - self.par.eta * self.grad.mean(axis=0)
            self.w = np.where(self.w < 0, np.zeros_like(self.w), self.w)
        else:
            self.w = self.w - self.par.eta * self.grad.mean(axis=0)

        
# ---------------------------------------------------------------------------

class NeuronClassPyTorch(nn.Module):

    def __init__(self,par):
        super(NeuronClassPyTorch,self).__init__() 
        """
        Args:
        - par: object containing the parameters of the neuron
        """
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)              
        
        # 初始化二维坐标位置编码相关参数
        # 神经元自身位置 (x, y)
        self.neuron_position = getattr(par, 'neuron_position', (0.0, 0.0))  # 神经元在地图中的位置
        # 目标奖励位置 (x, y)
        self.target_position = getattr(par, 'target_position', (0.0, 0.0))  # 目标奖励位置
        # 奖励参数
        self.reward_strength = getattr(par, 'reward_strength', 1.0)  # 奖励强度
        self.reward_sigma = getattr(par, 'reward_sigma', 1.0)  # 奖励分布的标准差
        # 异序列抑制参数
        self.mu = getattr(par, 'mu', 0.0)  # 异序列抑制强度，默认0
        self.gamma = getattr(par, 'gamma', 0.95)  # 时序迹更新系数
        self.sigma_p = getattr(par, 'sigma_p', 0.15)  # 资格迹差异容限

    def initialize(self):
        """
        supported initialization methods:
        - 'trunc_gauss': Initialize the weights using a truncated Gaussian distribution.
        - 'uniform': Initialize the weights using a uniform distribution.
        - 'fixed': Initialize the weights with a fixed value.
        """
        if self.par.init == 'trunc_gauss':
            self.w = nn.Parameter(torch.empty(self.par.N)).to(self.par.device)
            torch.nn.init.trunc_normal_(self.w, mean=self.par.init_mean, std=1/np.sqrt(self.par.N),
                                    a=self.par.init_a,b=self.par.init_b)
        elif self.par.init == 'uniform':
            self.w = nn.Parameter(self.par.init_mean*torch.rand(self.par.N)).to(self.par.device)
        elif self.par.init == 'fixed':
            self.w = nn.Parameter(self.par.init_mean*torch.ones(self.par.N)).to(self.par.device)
        else:
            self.w = nn.Parameter(torch.empty(self.par.N)).to(self.par.device)
            torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.N))
        
        # 初始化时序迹（非参数，用于记录统计信息）
        self.xi = torch.zeros(self.par.N).to(self.par.device)  # 时序迹，记录资格迹的统计平均模式
        
    def state(self):
        """
        state variables:
        - self.v: Tensor of shape (self.par.batch,) representing the voltage.
        - self.z: Tensor of shape (self.par.batch,) representing the output spikes.
        - self.p: Tensor of shape (self.par.batch, self.par.N) representing the eligibiliy traces.
        - self.epsilon: Tensor of shape (self.par.batch, self.par.N) representing the prediction errors.
        - self.E: Tensor of shape (self.par.batch,) representing the global signal.
        - self.grad: Tensor of shape (self.par.batch, self.par.N) representing the gradient.
        - self.R: Tensor of shape (self.par.batch,) representing the reward signal.
        - self.Gamma: Tensor of shape (self.par.batch,) representing the sequence consistency score.
        """
        self.v = torch.zeros(self.par.batch).to(self.par.device)
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.p = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.epsilon = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.E = torch.zeros(self.par.batch).to(self.par.device)
        self.grad = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.R = torch.ones(self.par.batch).to(self.par.device)  # 默认为1，表示无奖励调制
        self.Gamma = torch.ones(self.par.batch).to(self.par.device)  # 默认为1，表示完全匹配
        
    def __call__(self,x):
        """
        Parameters:
        x : Tensor of shape (self.par.batch, self.par.N)
        
        Updates:
        - self.v: Tensor of shape (self.par.batch,) 
        - self.z: Tensor of shape (self.par.batch,)
        """
        self.v = self.alpha*self.v + x@self.w \
                    - self.par.v_th*self.z.detach()
        
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.z[self.v - self.par.v_th > 0] = 1
        
    def backward_online(self,x):
        """
        Args:
        x : Tensor of shape (self.par.batch, self.par.N)
    
        Updates:
        - self.epsilon: Tensor of shape (self.par.batch, self.par.N) 
        - self.E: Tensor of shape (self.par.batch,) 
        - self.grad: Tensor of shape (self.par.batch, self.par.N) 
        - self.p: Tensor of shape (self.par.batch, self.par.N)
        - self.R: Tensor of shape (self.par.batch,) - 奖励信号
        - self.Gamma: Tensor of shape (self.par.batch,) - 序列一致性评分
        - self.xi: Tensor of shape (self.par.N) - 更新时序迹
        """
        self.epsilon =  x - self.v[:,None]*self.w[None,:]
        self.E = self.epsilon@self.w
        
        # 计算基于位置的奖励信号
        self._compute_reward()
        
        # 计算基础梯度（不包含异序列抑制项）
        base_grad = -(self.v[:,None]*self.epsilon + \
                        self.E[:,None]*self.p)
        
        # 先更新资格迹
        self.p = self.alpha*self.p + x
        
        # 更新时序迹：使用资格迹的平均值作为当前时刻的资格迹
        avg_p = self.p.mean(dim=0)  # 计算批次平均的资格迹
        self.xi = self.gamma * self.xi + (1 - self.gamma) * avg_p  # 更新时序迹
        
        # 计算序列一致性评分Gamma
        # 默认完全匹配
        self.Gamma = torch.ones(self.par.batch).to(self.par.device)
        if self.mu > 0:  # 只有当mu>0时才计算一致性评分
            # 对每个批次样本，计算当前p与xi的相似性
            similarities = torch.exp(-0.5 * ((self.p - self.xi[None, :]) / self.sigma_p) ** 2)
            self.Gamma = similarities.mean(dim=1)  # 计算每个批次的平均相似度
        
        # 添加异序列抑制项（当mu>0时）
        if self.mu > 0:
            # 异序列抑制项：mu*(1-Gamma)*p
            suppression_term = self.mu * (1 - self.Gamma)[:, None] * self.p
            base_grad += suppression_term
        
        # 使用奖励信号调制梯度
        self.grad = base_grad * self.R[:, None]  # 应用奖励调制
    
    def _compute_reward(self):
        """
        计算基于二维坐标的奖励信号
        神经元放电时，根据其在地图中的绝对位置与目标位置的距离给予奖励
        """
        # 计算二维欧几里得距离
        dx = self.neuron_position[0] - self.target_position[0]
        dy = self.neuron_position[1] - self.target_position[1]
        distance = torch.sqrt(dx**2 + dy**2)
        
        # 使用高斯函数计算奖励：距离越近，奖励越大
        # 奖励范围在1到1+reward_strength之间
        self.R = 1.0 + self.reward_strength * torch.exp(-0.5 * (distance / self.reward_sigma) ** 2)
        
        # 如果神经元发放了，增强奖励效果
        self.R = torch.where(self.z > 0, self.R * 1.5, self.R)
        
    def update_online(self):
        """
        更新权重，通过设置权重的梯度，使其能够随着 loss 梯度下降
        
        'soft': 权重 (w) 的梯度与权重值成正比，乘以学习率 (eta) 和梯度沿轴 0 的均值
        
        'hard': 权重 (w) 的梯度为学习率 (eta) 乘以梯度沿轴 0 的均值
                在优化器更新后，负权重将被置零
        
        'else': 权重 (w) 的梯度为学习率 (eta) 乘以梯度沿轴 0 的均值
        """
        # 确保权重参数 requires_grad 为 True
        if not self.w.requires_grad:
            self.w.requires_grad = True
        
        # 根据不同的约束类型设置权重的梯度
        if self.par.bound == 'soft':
            # soft 约束：梯度与权重值成正比
            self.w.grad = -self.w * (self.par.eta * torch.mean(self.grad, dim=0))
        elif self.par.bound == 'hard':
            # hard 约束：标准梯度，在优化器更新后会将负权重置零
            self.w.grad = -self.par.eta * torch.mean(self.grad, dim=0)
        else:
            # 默认：标准梯度下降
            self.w.grad = -self.par.eta * torch.mean(self.grad, dim=0)
            
    def apply_weight_constraints(self):
        """
        应用权重约束，在优化器更新权重后调用
        
        主要用于处理 'hard' 约束，将负权重置零
        """
        if hasattr(self.par, 'bound') and self.par.bound == 'hard':
            with torch.no_grad():
                self.w.data = torch.where(self.w.data < 0, torch.zeros_like(self.w.data), self.w.data)
