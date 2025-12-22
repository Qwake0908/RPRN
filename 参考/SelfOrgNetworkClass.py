
import numpy as np
import math
import torch
import torch.nn as nn

class NetworkClassNumPy():
    
    def __init__(self,par):
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)  
        self.beta = (1-self.par.dt/self.par.tau_x)    
        
        # 异序列抑制相关参数
        self.mu = getattr(self.par, 'mu', 0.0)  # 异序列抑制强度，默认0
        self.gamma = getattr(self.par, 'gamma', 0.95)  # 时序迹更新系数
        self.sigma_p = getattr(self.par, 'sigma_p', 0.15)  # 资格迹差异容限
        
        # 初始化二维坐标相关参数
        # 地图大小 - 默认是10x10的网格
        self.map_width = getattr(par, 'map_width', 10.0)
        self.map_height = getattr(par, 'map_height', 10.0)
        
        # 目标奖励位置 (x, y)
        self.target_position = getattr(par, 'target_position', (5.0, 5.0))
        
        # 神经元二维坐标位置数组
        self.neuron_positions = None

    def initialize(self):
        """
        初始化权重矩阵和神经元的二维坐标位置
        为每个神经元分配固定的二维坐标位置，形成一张地图
        支持一维和二维位置排列
        """
        # 初始化权重矩阵
        self.w = np.zeros((self.par.N,self.par.nn))
        self.xi = np.zeros((self.par.N, self.par.nn))  # 初始化时序迹，记录资格迹的统计平均模式

        if self.par.network_type == 'nearest':
            self.w[:self.par.n_in,:] = self.par.init_mean*np.ones((self.par.n_in,self.par.nn))
            self.w[self.par.n_in:,:] = self.par.init_rec*np.ones((2,self.par.nn))

        if self.par.network_type == 'all':
            self.w[:self.par.n_in,:] = self.par.init_mean*np.ones((self.par.n_in,self.par.nn))
            self.w[self.par.n_in:,:] = self.par.init_rec*np.ones((self.par.nn,self.par.nn))
        
        if self.par.network_type == 'random':
            self.w[:self.par.N_in,:] = np.random.uniform(0.,self.par.init_mean,(self.par.N_in,self.par.nn))
            self.w[self.par.N_in:,:] = np.random.uniform(0.,self.par.init_rec,(self.par.nn,self.par.nn))
        
        # 为每个神经元分配二维坐标位置
        # 检查是否需要一维排列
        use_1d_arrangement = getattr(self.par, 'use_1d_arrangement', False)
        
        if use_1d_arrangement:
            # 一维排列：所有神经元在一条水平线上均匀分布
            self.neuron_positions = np.zeros((self.par.nn, 2))
            spacing = self.map_width / (self.par.nn - 1) if self.par.nn > 1 else 0
            
            for i in range(self.par.nn):
                self.neuron_positions[i, 0] = i * spacing  # x坐标
                self.neuron_positions[i, 1] = self.map_height / 2  # y坐标（居中）
        else:
            # 二维网格排列（默认）
            # 计算网格大小以均匀分布神经元
            grid_size = math.ceil(math.sqrt(self.par.nn))
            cell_width = self.map_width / (grid_size - 1) if grid_size > 1 else self.map_width
            cell_height = self.map_height / (grid_size - 1) if grid_size > 1 else self.map_height
            
            # 生成神经元位置数组
            self.neuron_positions = np.zeros((self.par.nn, 2))
            for i in range(self.par.nn):
                row = i // grid_size
                col = i % grid_size
                self.neuron_positions[i, 0] = col * cell_width  # x坐标
                self.neuron_positions[i, 1] = row * cell_height  # y坐标

    def state(self):
        """
        初始化网络状态变量
        """
        self.v = np.zeros((self.par.batch,self.par.nn))
        self.z = np.zeros((self.par.batch,self.par.nn))
        self.z_out = np.zeros((self.par.batch,self.par.nn))
        
        self.p = np.zeros((self.par.batch,self.par.N,self.par.nn))
        self.epsilon = np.zeros((self.par.batch,self.par.N,self.par.nn))
        self.grad = np.zeros((self.par.batch,self.par.N,self.par.nn))
        self.Gamma = np.ones((self.par.batch, self.par.nn))  # 序列一致性评分，默认为1
        
        # 奖励信号 - 默认为1，表示无奖励调制
        self.R = np.ones((self.par.batch, self.par.nn))

    def __call__(self,x):
        """
        前向传播函数，计算神经元电压和输出脉冲
        并在神经元发放后计算基于位置的奖励
        """
        self._get_inputs(x)
        
        self.v = self.alpha*self.v + np.einsum('ijk,jk->ik',self.x,self.w) \
                    - self.par.v_th*self.z
        
        self.z = np.zeros((self.par.batch,self.par.nn))
        self.z[self.v - self.par.v_th > 0] = 1
        
        # 计算基于位置的奖励
        self._compute_reward()

    def backward_online(self,x):
        """
        计算在线学习的梯度，使用奖励信号调制权重更新
        """
        self._get_inputs(x)
        
        self.epsilon =  self.x - np.einsum('ij,kj->ikj',self.v,self.w)
        self.heterosyn = np.einsum('ikj,kj->ij',self.epsilon,self.w)
        self.grad = - np.einsum('ij,ikj->ikj',self.v,self.epsilon) \
                    - np.einsum('ij,ikj->ikj', self.heterosyn,self.p)
        
        # 先更新资格迹
        self.p = self.alpha*self.p + self.x
        
        # 添加异序列抑制项（当mu>0时）
        if self.mu > 0:
            # 更新时序迹：使用资格迹的平均值作为当前时刻的资格迹
            avg_p = self.p.mean(axis=0)  # 计算批次平均的资格迹
            self.xi = self.gamma * self.xi + (1 - self.gamma) * avg_p  # 更新时序迹
            
            # 计算序列一致性评分Gamma
            for b in range(self.par.batch):
                for n in range(self.par.nn):
                    # 计算每个突触的资格迹与时序迹的高斯相似度
                    similarities = np.exp(-0.5 * ((self.p[b, :, n] - self.xi[:, n]) / self.sigma_p) ** 2)
                    # 计算平均相似度作为Gamma
                    self.Gamma[b, n] = np.mean(similarities)
            
            # 添加异序列抑制项：mu*(1-Gamma)*p
            suppression_term = self.mu * np.einsum('ij,ikj->ikj', (1 - self.Gamma), self.p)
            self.grad = self.grad + suppression_term
        
        # 应用奖励信号调制梯度（将所有梯度项，包括异序列抑制项，都乘以奖励）
        # R的形状是(batch, nn)，需要扩展为(batch, 1, nn)以匹配grad的形状(batch, N, nn)
        self.grad = self.grad * self.R[:, None, :]

    def update_online(self):

        if self.par.bound == 'soft':
            self.w = self.w - self.w*self.par.eta*self.grad.mean(axis=0)
        elif self.par.bound == 'hard':
            self.w = self.w - self.par.eta*self.grad.mean(axis=0)
            self.w = np.where(self.w<0,np.zeros_like(self.w),self.w)
        else: self.w = self.w - self.par.eta*self.grad.mean(axis=0)


    def _get_inputs(self, x):
        """
        根据网络类型组织输入信号
        """
        self.z_out = self.beta*self.z_out + self.z
        self.x = np.zeros((self.par.batch,self.par.N,self.par.nn))
    
        if self.par.network_type == 'nearest':

            for n in range(self.par.nn): 
                if n == 0:
                    self.x[:,:,n] = np.hstack((x[:,:,n],
                                                 np.zeros((self.par.batch,1)),
                                                 self.z_out[:,n+1][:,None]))
                elif n == self.par.nn - 1:
                    self.x[:,:,n] = np.hstack((x[:,:,n],
                                                 self.z_out[:,n-1][:,None],
                                                 np.zeros((self.par.batch,1))))
                else:
                    self.x[:,:,n] = np.hstack((x[:, :, n],
                                                 self.z_out[:,n-1][:,None],
                                                 self.z_out[:,n+1][:,None]))
        else:
            for n in range(self.par.nn):
                temp_z_out = np.delete(self.z_out, n, axis=1)
                self.x[:,:,n] = np.hstack((x[:,:,n], temp_z_out, np.zeros((self.par.batch,1))))
    
    def _compute_reward(self):
        """
        计算基于二维坐标的奖励信号
        根据神经元的绝对位置与目标位置的距离给予奖励
        """
        # 获取奖励相关参数
        reward_strength = getattr(self.par, 'reward_strength', 1.0)
        reward_sigma = getattr(self.par, 'reward_sigma', 1.0)
        
        # 初始化奖励为1（无奖励调制状态）
        self.R = np.ones((self.par.batch, self.par.nn))
        
        # 对每个神经元计算其到目标位置的距离
        for n in range(self.par.nn):
            # 获取神经元的位置
            neuron_x, neuron_y = self.neuron_positions[n]
            # 获取目标位置
            target_x, target_y = self.target_position
            
            # 计算欧几里得距离
            dx = neuron_x - target_x
            dy = neuron_y - target_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # 使用高斯函数计算奖励：距离越近，奖励越大
            # 奖励范围在1到1+reward_strength之间
            reward = 1.0 + reward_strength * np.exp(-0.5 * (distance / reward_sigma) ** 2)
            
            # 对所有批次设置相同的奖励值（因为神经元位置是固定的）
            self.R[:, n] = reward
        
        # 对于发放的神经元，增强奖励效果
        self.R = np.where(self.z > 0, self.R * 1.5, self.R)

#---------------------------------------------------------------------------

class NetworkClassPyTorch(nn.Module):
    
    def __init__(self, par):
        super(NetworkClassPyTorch, self).__init__()
        self.par = par
        self.alpha = (1 - self.par.dt / self.par.tau_m)
        self.beta = (1 - self.par.dt / self.par.tau_x)
        
        # 异序列抑制相关参数
        self.mu = getattr(self.par, 'mu', 0.0)  # 异序列抑制强度，默认0
        self.gamma = getattr(self.par, 'gamma', 0.95)  # 时序迹更新系数
        self.sigma_p = getattr(self.par, 'sigma_p', 0.15)  # 资格迹差异容限
        
        # 初始化二维坐标相关参数
        # 地图大小 - 默认是10x10的网格
        self.map_width = getattr(par, 'map_width', 10.0)
        self.map_height = getattr(par, 'map_height', 10.0)
        
        # 目标奖励位置 (x, y)
        self.target_position = getattr(par, 'target_position', (5.0, 5.0))
        
        # 神经元二维坐标位置数组
        self.neuron_positions = None
    
    def initialize(self):
        """
        初始化权重矩阵和神经元的二维坐标位置
        为每个神经元分配固定的二维坐标位置，形成一张地图
        """
        # 初始化权重矩阵
        self.w = nn.Parameter(torch.zeros((self.par.N, self.par.nn))).to(self.par.device)
        self.xi = torch.zeros((self.par.N, self.par.nn)).to(self.par.device)  # 初始化时序迹，记录资格迹的统计平均模式
        
        if self.par.network_type == 'nearest':
            self.w.data[:self.par.n_in, :] = self.par.init_mean * torch.ones((self.par.n_in, self.par.nn)).to(self.par.device)
            self.w.data[self.par.n_in:, :] = self.par.init_rec * torch.ones((2, self.par.nn)).to(self.par.device)
        
        if self.par.network_type == 'all':
            self.w.data[:self.par.n_in, :] = self.par.init_mean * torch.ones((self.par.n_in, self.par.nn)).to(self.par.device)
            self.w.data[self.par.n_in:, :] = self.par.init_rec * torch.ones((self.par.nn, self.par.nn)).to(self.par.device)
        
        # 为每个神经元分配二维坐标位置
        # 计算网格大小以均匀分布神经元
        grid_size = math.ceil(math.sqrt(self.par.nn))
        cell_width = self.map_width / (grid_size - 1) if grid_size > 1 else self.map_width
        cell_height = self.map_height / (grid_size - 1) if grid_size > 1 else self.map_height
        
        # 生成神经元位置数组
        self.neuron_positions = np.zeros((self.par.nn, 2))
        for i in range(self.par.nn):
            row = i // grid_size
            col = i % grid_size
            self.neuron_positions[i, 0] = col * cell_width  # x坐标
            self.neuron_positions[i, 1] = row * cell_height  # y坐标
    
    def state(self):
        """
        初始化网络状态变量
        """
        self.v = torch.zeros((self.par.batch, self.par.nn)).to(self.par.device)
        self.z = torch.zeros((self.par.batch, self.par.nn)).to(self.par.device)
        self.z_out = torch.zeros((self.par.batch, self.par.nn)).to(self.par.device)
        
        self.p = torch.zeros((self.par.batch, self.par.N, self.par.nn)).to(self.par.device)
        self.epsilon = torch.zeros((self.par.batch, self.par.N, self.par.nn)).to(self.par.device)
        self.grad = torch.zeros((self.par.batch, self.par.N, self.par.nn)).to(self.par.device)
        self.heterosyn = torch.zeros((self.par.batch, self.par.nn)).to(self.par.device)
        self.Gamma = torch.ones((self.par.batch, self.par.nn)).to(self.par.device)  # 序列一致性评分，默认为1
        
        # 奖励信号 - 默认为1，表示无奖励调制
        self.R = torch.ones((self.par.batch, self.par.nn)).to(self.par.device)
    
    def __call__(self, x):
        """
        前向传播函数，计算神经元电压和输出脉冲
        并在神经元发放后计算基于位置的奖励
        """
        self._get_inputs(x)
        
        self.v = self.alpha * self.v + torch.einsum('ijk,jk->ik', self.x, self.w) - \
                self.par.v_th * self.z
        
        self.z = torch.zeros((self.par.batch, self.par.nn)).to(self.par.device)
        self.z[self.v - self.par.v_th > 0] = 1
        
        # 计算基于位置的奖励
        self._compute_reward()
    
    def backward_online(self, x):
        """
        计算在线学习的梯度，使用奖励信号调制权重更新
        """
        self._get_inputs(x)
        
        self.epsilon = self.x - torch.einsum('ij,kj->ikj', self.v, self.w)
        self.heterosyn = torch.einsum('ikj,kj->ij', self.epsilon, self.w)
        self.grad = - torch.einsum('ij,ikj->ikj', self.v, self.epsilon) - \
                    torch.einsum('ij,ikj->ikj', self.heterosyn, self.p)
        
        # 先更新资格迹
        self.p = self.alpha * self.p + self.x
        
        # 添加异序列抑制项（当mu>0时）
        if self.mu > 0:
            # 更新时序迹：使用资格迹的平均值作为当前时刻的资格迹
            avg_p = self.p.mean(dim=0)  # 计算批次平均的资格迹
            self.xi = self.gamma * self.xi + (1 - self.gamma) * avg_p  # 更新时序迹
            
            # 计算序列一致性评分Gamma
            similarities = torch.exp(-0.5 * ((self.p - self.xi[None, :, :]) / self.sigma_p) ** 2)
            self.Gamma = similarities.mean(dim=1)  # 计算每个批次的平均相似度
            
            # 添加异序列抑制项：mu*(1-Gamma)*p
            suppression_term = self.mu * (1 - self.Gamma)[:, None, :] * self.p
            self.grad = self.grad + suppression_term
        
        # 应用奖励信号调制梯度（将所有梯度项，包括异序列抑制项，都乘以奖励）
        # R的形状是(batch, nn)，需要扩展为(batch, 1, nn)以匹配grad的形状(batch, N, nn)
        self.grad = self.grad * self.R[:, None, :]
    
    def update_online(self):
        with torch.no_grad():
            if self.par.bound == 'soft':
                self.w.data = self.w.data - self.w.data * self.par.eta * torch.mean(self.grad, dim=0)
            elif self.par.bound == 'hard':
                self.w.data = self.w.data - self.par.eta * torch.mean(self.grad, dim=0)
                self.w.data = torch.where(self.w.data < 0, torch.zeros_like(self.w.data), self.w.data)
            else:
                self.w.data = self.w.data - self.par.eta * torch.mean(self.grad, dim=0)
    
    def _get_inputs(self, x):
        """
        根据网络类型组织输入信号
        """
        self.z_out = self.beta * self.z_out + self.z
        self.x = torch.zeros((self.par.batch, self.par.N, self.par.nn)).to(self.par.device)
        
        if self.par.network_type == 'nearest':
            for n in range(self.par.nn):
                if n == 0:
                    zeros_col = torch.zeros((self.par.batch, 1)).to(self.par.device)
                    self.x[:, :, n] = torch.cat((x[:, :, n], zeros_col, self.z_out[:, n+1].unsqueeze(1)), dim=1)
                elif n == self.par.nn - 1:
                    zeros_col = torch.zeros((self.par.batch, 1)).to(self.par.device)
                    self.x[:, :, n] = torch.cat((x[:, :, n], self.z_out[:, n-1].unsqueeze(1), zeros_col), dim=1)
                else:
                    self.x[:, :, n] = torch.cat((x[:, :, n], self.z_out[:, n-1].unsqueeze(1), self.z_out[:, n+1].unsqueeze(1)), dim=1)
        else:
            for n in range(self.par.nn):
                temp_z_out = torch.cat((self.z_out[:, :n], self.z_out[:, n+1:]), dim=1)
                zeros_col = torch.zeros((self.par.batch, 1)).to(self.par.device)
                self.x[:, :, n] = torch.cat((x[:, :, n], temp_z_out, zeros_col), dim=1)
    
    def _compute_reward(self):
        """
        计算基于二维坐标的奖励信号
        根据神经元的绝对位置与目标位置的距离给予奖励
        """
        # 获取奖励相关参数
        reward_strength = getattr(self.par, 'reward_strength', 1.0)
        reward_sigma = getattr(self.par, 'reward_sigma', 1.0)
        
        # 初始化奖励为1（无奖励调制状态）
        self.R = torch.ones((self.par.batch, self.par.nn)).to(self.par.device)
        
        # 对每个神经元计算其到目标位置的距离
        for n in range(self.par.nn):
            # 获取神经元的位置
            neuron_x, neuron_y = self.neuron_positions[n]
            # 获取目标位置
            target_x, target_y = self.target_position
            
            # 计算欧几里得距离
            dx = neuron_x - target_x
            dy = neuron_y - target_y
            distance = math.sqrt(dx**2 + dy**2)
            
            # 使用高斯函数计算奖励：距离越近，奖励越大
            # 奖励范围在1到1+reward_strength之间
            reward = 1.0 + reward_strength * math.exp(-0.5 * (distance / reward_sigma) ** 2)
            
            # 对所有批次设置相同的奖励值（因为神经元位置是固定的）
            self.R[:, n] = reward
        
        # 对于发放的神经元，增强奖励效果
        self.R = torch.where(self.z > 0, self.R * 1.5, self.R)

    def apply_weight_constraints(self):
        """
        应用权重约束
        """
        with torch.no_grad():
            if self.par.bound == 'hard':
                self.w.data = torch.where(self.w.data < 0, torch.zeros_like(self.w.data), self.w.data)