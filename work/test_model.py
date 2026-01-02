import sys
import os

# 添加项目根目录到Python路径，确保能导入根目录下的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.rpr_network import RPRNetwork
from utils.loss import RDPLLoss
from data.data import SpikeDataset
from scripts.trainer import Trainer
import torch.optim as optim


def test_rpr_model():
    """测试RPR模型"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建数据集
    dataset = SpikeDataset(n_samples=100, time_steps=100, n_neurons=10, batch_size=32, device=device)
    
    # 创建模型（1d连接模式）
    layer_sizes = [10, 20, 10]
    model = RPRNetwork(
        layer_sizes,
        connection_mode='1d',
        device=device
    )
    
    # 定义损失函数和优化器
    loss_fn = RDPLLoss(mu=0.0)  # 使用修改后的默认值
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建训练器
    trainer = Trainer(model, loss_fn, optimizer, device=device)
    
    # 训练模型
    print("Starting training...")
    trainer.train(dataset, epochs=3, batch_size=32)
    
    # 保存模型到save文件夹
    trainer.save_model("rpr_model_test.pth")
    
    # 评估模型
    print("\nEvaluating model...")
    trainer.evaluate(dataset)
    
    # 测试模型加载
    print("\nTesting model loading...")
    new_model = RPRNetwork(layer_sizes, connection_mode='1d', device=device)
    new_loss_fn = RDPLLoss()
    new_optimizer = optim.Adam(new_model.parameters(), lr=1e-3)
    new_trainer = Trainer(new_model, new_loss_fn, new_optimizer, device=device)
    # 从save文件夹加载模型
    new_trainer.load_model("save/rpr_model_test.pth")
    
    print("Model testing completed successfully!")


if __name__ == "__main__":
    test_rpr_model()
