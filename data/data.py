import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SpikeDataset(Dataset):
    """时序脉冲数据集"""
    def __init__(self, n_samples=100, time_steps=100, n_neurons=10, 
                 freq=0.1, batch_size=32, device='cpu'):
        """初始化数据集
        n_samples: 样本数量
        time_steps: 时间步数（1ms步长）
        n_neurons: 神经元数量
        freq: 基础脉冲频率
        batch_size: 批次大小
        """
        self.n_samples = n_samples
        self.time_steps = time_steps
        self.n_neurons = n_neurons
        self.freq = freq
        self.batch_size = batch_size
        self.device = device
        
        # 生成数据
        self.data, self.rewards = self.generate_data()
    
    def generate_data(self):
        """生成时序脉冲数据和奖励信号"""
        data = []
        rewards = []
        
        for _ in range(self.n_samples):
            # 生成随机脉冲序列
            spike = torch.rand(self.time_steps, self.n_neurons) < self.freq
            spike = spike.float()
            
            # 生成奖励信号：与神经元相关，默认全为1
            # R_{t,k}：针对不同时刻t和不同神经元k的奖励信号
            reward = torch.ones(self.time_steps, self.n_neurons)
            
            data.append(spike)
            rewards.append(reward)
        
        # 转换为张量
        data = torch.stack(data, dim=0).to(self.device)
        rewards = torch.stack(rewards, dim=0).to(self.device)
        
        return data, rewards
    
    def generate_patterned_data(self, pattern_length=5):
        """生成具有特定模式的脉冲序列"""
        data = []
        rewards = []
        
        for _ in range(self.n_samples):
            spike = torch.zeros(self.time_steps, self.n_neurons)
            
            # 在随机位置插入模式
            for _ in range(3):
                start = torch.randint(0, self.time_steps - pattern_length, (1,)).item()
                for t in range(pattern_length):
                    # 生成交替激活的模式
                    neuron_idx = t % self.n_neurons
                    spike[start + t, neuron_idx] = 1.0
            
            # 添加随机背景脉冲
            background = torch.rand(self.time_steps, self.n_neurons) < self.freq * 0.5
            spike += background.float()
            spike = (spike > 0).float()
            
            # 生成奖励信号：与神经元相关，默认全为1
            # R_{t,k}：针对不同时刻t和不同神经元k的奖励信号
            reward = torch.ones(self.time_steps, self.n_neurons)
            
            data.append(spike)
            rewards.append(reward)
        
        data = torch.stack(data, dim=0).to(self.device)
        rewards = torch.stack(rewards, dim=0).to(self.device)
        
        return data, rewards
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """获取单个样本"""
        return self.data[idx], self.rewards[idx]
    
    def get_batch(self):
        """获取随机批次数据"""
        indices = torch.randint(0, self.n_samples, (self.batch_size,))
        return self.data[indices], self.rewards[indices]

class SpikeGenerator:
    """脉冲生成器"""
    
    @staticmethod
    def poisson_spike(rate, time_steps, dt=1.0):
        """生成泊松分布的脉冲序列
        rate: 发放率（Hz）
        time_steps: 时间步数
        dt: 时间步长（ms）
        """
        return torch.rand(time_steps) < (rate * dt / 1000)
    
    @staticmethod
    def periodic_spike(period, time_steps, phase=0):
        """生成周期性脉冲序列
        period: 周期（时间步数）
        time_steps: 时间步数
        phase: 相位偏移
        """
        return (torch.arange(time_steps) + phase) % period == 0
    
    @staticmethod
    def burst_spike(burst_rate, burst_duration, inter_burst_interval, time_steps):
        """生成爆发式脉冲序列
        burst_rate: 爆发期间的发放率
        burst_duration: 爆发持续时间
        inter_burst_interval: 爆发间隔
        time_steps: 时间步数
        """
        spike = torch.zeros(time_steps)
        current_time = 0
        
        while current_time < time_steps:
            # 爆发期
            burst_end = min(current_time + burst_duration, time_steps)
            spike[current_time:burst_end] = torch.rand(burst_end - current_time) < burst_rate
            
            # 间隔期
            current_time = burst_end + inter_burst_interval
        
        return spike.float()
