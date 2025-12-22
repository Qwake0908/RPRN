import torch
import torch.optim as optim
from rpr_network import RPRNetwork
from loss import RDPLLoss
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
        x_batch: [batch_size, time_steps, n_neurons]
        reward_batch: [batch_size, time_steps]
        """
        self.optimizer.zero_grad()
        
        # 前向传播 - 现在只返回outputs, gamma_t, p
        outputs, gamma_t, p = self.model(x_batch, gamma, sigma)
        
        # 计算预测值
        # 这里假设模型预测下一个时间步的输入
        x_hat = x_batch[:, 1:, :]
        x = x_batch[:, :-1, :]
        
        # 调整奖励信号和一致性评分的维度
        reward = reward_batch[:, :-1]
        gamma_t = gamma_t[:, :-1]
        p = p[:, :-1, :]
        
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
    
    def save_model(self, path):
        """保存模型参数
        path: 保存路径
        """
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

# 示例用法
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建数据集
    dataset = SpikeDataset(n_samples=100, time_steps=100, n_neurons=10, batch_size=32, device=device)
    
    # 创建模型
    layer_sizes = [10, 20, 10]  # 输入层, 隐藏层, 输出层
    model = RPRNetwork(layer_sizes, connection_mode='1d', device=device)
    
    # 定义损失函数和优化器
    loss_fn = RDPLLoss(mu=0.1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建训练器
    trainer = Trainer(model, loss_fn, optimizer, device=device)
    
    # 训练模型
    trainer.train(dataset, epochs=10, batch_size=32)
    
    # 保存模型
    trainer.save_model("rpr_model.pth")
    
    # 评估模型
    trainer.evaluate(dataset)
