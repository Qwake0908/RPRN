import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from models.rpr_network import RPRNetwork
from utils.loss import RDPLLoss
from data.data import SpikeDataset, SpikeGenerator
from scripts.trainer import Trainer
import torch.optim as optim


def create_paper_network():
    """创建论文中描述的10神经元网络
    网络中的每个神经元都接收来自8个突触前神经元的输入
    第一层应该有80个神经元（10个神经元×8个连接）
    使用1d连接模式，实现最近邻递归连接
    """
    # 修改第一层神经元数量为80个（10个神经元×8个连接）
    layer_sizes = [80, 10]  # 输入层80个神经元，输出层10个神经元
    
    # 使用1d连接模式，自动实现最近邻递归连接
    model = RPRNetwork(layer_sizes, connection_mode='1d', device='cuda')
    
    return model


def generate_sequence_data():
    """生成论文中描述的数据
    网络中的每个神经元都接收来自8个突触前神经元的输入
    这些神经元以2毫秒的相对延迟依次放电
    产生总长度为16毫秒的序列
    """
    time_steps = 100  # 总时间步长（1ms步长）
    n_neurons = 8  # 每个神经元接收来自8个突触前神经元的输入
    
    # 生成具有2毫秒相对延迟的序列（粉色尖峰模式）
    spike_pattern = torch.zeros(time_steps, n_neurons)
    
    # 8个神经元以2ms延迟依次放电，总长度16ms
    start_time = 30  # 序列开始时间
    for i in range(n_neurons):
        spike_time = start_time + i * 2
        if spike_time < time_steps:
            spike_pattern[spike_time, i] = 1.0
    
    return spike_pattern


def generate_network_input(spike_pattern):
    """生成网络的输入
    10个神经元，每个神经元接收来自8个突触前神经元的输入
    第一层有80个神经元（10×8）
    """
    time_steps = spike_pattern.shape[0]
    input_seq = torch.zeros(time_steps, 80)  # 80个输入神经元（10×8）
    
    # 为每个神经元的8个突触前输入设置放电模式
    for neuron_idx in range(10):
        start_time_offset = neuron_idx * 4  # 每个神经元的序列起始时间晚4ms
        
        for synapse_idx in range(8):
            # 计算该突触前神经元在80个输入神经元中的位置
            input_idx = neuron_idx * 8 + synapse_idx
            
            for t in range(time_steps):
                # 设置第neuron_idx个神经元的第synapse_idx个突触前输入
                if t >= start_time_offset and t - start_time_offset < spike_pattern.shape[0]:
                    # 使用spike_pattern中的对应突触前神经元放电模式
                    input_seq[t, input_idx] = spike_pattern[t - start_time_offset, synapse_idx]
    
    return input_seq


def train_paper_network():
    """训练论文中描述的网络
    重复2000次训练
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建网络
    model = create_paper_network()
    
    # 定义损失函数和优化器
    loss_fn = RDPLLoss(mu=0.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建训练器
    trainer = Trainer(model, loss_fn, optimizer, device=device)
    
    # 生成基本序列
    base_pattern = generate_sequence_data()
    print(f"Base pattern shape: {base_pattern.shape}")
    
    # 创建训练数据集
    # 生成包含80个输入神经元的输入序列
    input_seq = generate_network_input(base_pattern)
    # 将输入序列移到正确的设备上
    input_seq = input_seq.to(device)
    
    # 复制多次以创建多个样本
    train_data = input_seq.unsqueeze(0)  # [1, 100, 80]
    
    # 生成奖励信号：默认全为1
    # 维度：[batch_size, time_steps, n_outputs]
    rewards = torch.ones(1, 100, 10, device=device)
    
    # 训练网络
    print("Starting training...")
    epochs = 200  # 调整为200个epoch
    for epoch in range(epochs):
        # 使用训练数据进行训练
        # train_data已经是3维张量：[1, 100, 80]
        loss = trainer.train_step(train_data, rewards)
        
        # 每20个epoch打印一次
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
    
    # 保存训练后的模型
    trainer.save_model("paper_model_trained.pth")
    print("Training completed!")
    
    return model


def test_recall(model):
    """测试网络的回忆功能
    """
    # 生成测试数据：80个输入神经元
    base_pattern = generate_sequence_data()
    test_input = generate_network_input(base_pattern)
    # 确保测试数据在与模型相同的设备上
    test_input = test_input.to(next(model.parameters()).device)
    test_input = test_input.unsqueeze(0)  # [1, 100, 80]
    
    # 测试前向传播
    outputs, gamma_t, p = model(test_input)
    
    # 打印结果
    print("\nRecall test results:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {outputs.shape}")
    
    # 统计每个神经元的放电次数
    for i in range(10):
        spike_count = outputs[0, :, i].sum().item()
        print(f"Neuron {i+1} spike count: {spike_count}")
    
    return outputs


def main():
    """主函数"""
    # 训练网络
    model = train_paper_network()
    
    # 测试回忆功能
    outputs = test_recall(model)
    
    # 保存结果
    torch.save(outputs, "save/paper_recall_results.pt")
    print("\nResults saved to save/paper_recall_results.pt")
    
    print("\nPaper network reproduction completed successfully!")


if __name__ == "__main__":
    main()
