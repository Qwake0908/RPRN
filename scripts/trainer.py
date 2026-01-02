import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from models.rpr_network import RPRNetwork
from utils.loss import RDPLLoss
from data.data import SpikeDataset, DataLoader

class Trainer:
    """RPR模型训练器"""
    def __init__(self, model, loss_fn, optimizer, device='cpu'):
        """初始化训练器
        model: RPRNetwork模型
        loss_fn: 损失函数
        optimizer: PyTorch优化器
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
        # 将模型移至设备
        self.model.to(self.device)
    
    def train_step(self, x_batch, reward_batch, gamma=0.9, sigma=0.1):
        """单步训练
        x_batch: [batch_size, time_steps, n_inputs]
        reward_batch: [batch_size, time_steps, n_outputs]
        """
        self.optimizer.zero_grad()
        
        # 前向传播 - 现在只返回outputs, gamma_t, p
        outputs, gamma_t, p = self.model(x_batch, gamma, sigma)
        
        # 计算预测值
        # 对于80输入10输出的模型，outputs是预测值
        x_hat = outputs
        x = torch.zeros_like(outputs)  # 目标值设为0，简化处理
        
        # 调整奖励信号和一致性评分的维度
        reward = reward_batch
        gamma_t = gamma_t
        p = p
        
        # 计算损失
        loss = self.loss_fn(x, x_hat, p, gamma_t, reward)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataset, epochs=100, batch_size=32, gamma=0.9, sigma=0.1):
        """训练循环
        dataset: SpikeDataset对象
        epochs: 训练轮数
        batch_size: 批次大小
        """
        self.model.train()
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for x_batch, reward_batch in dataloader:
                # 训练步骤
                loss = self.train_step(x_batch, reward_batch, gamma, sigma)
                total_loss += loss
                num_batches += 1
            
            # 计算平均损失
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def evaluate(self, dataset, batch_size=32, gamma=0.9, sigma=0.1):
        """评估模型
        dataset: SpikeDataset对象
        """
        self.model.eval()
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x_batch, reward_batch in dataloader:
                # 前向传播
                outputs, gamma_t, p = self.model(x_batch, gamma, sigma)
                
                x_hat = x_batch[:, 1:, :]
                x = x_batch[:, :-1, :]
                reward = reward_batch[:, :-1]
                gamma_t = gamma_t[:, :-1]
                p = p[:, :-1, :]
                
                loss = self.loss_fn(x, x_hat, p, gamma_t, reward)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Evaluation Loss: {avg_loss:.6f}")
        return avg_loss
    
    def save_model(self, filename):
        """保存模型参数
        filename: 保存文件名
        """
        # 确保save文件夹存在
        os.makedirs('save', exist_ok=True)
        # 构造完整路径
        path = os.path.join('save', filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """加载模型参数
        path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
