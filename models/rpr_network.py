import torch
import torch.nn as nn
from components.neuron import LIFNeuron

class RPRNetwork(nn.Module):
    def __init__(self, layer_sizes, connection_mode='1d', device='cpu'):
        """初始化RPR网络
        layer_sizes: 各层神经元数量列表，如[10, 20, 10]
        connection_mode: 连接模式，可选'1d', '2d', 'free'
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        self.connection_mode = connection_mode
        self.device = device
        self.num_layers = len(layer_sizes)
        
        # 初始化神经元层
        self.neuron_layers = nn.ModuleList()
        for size in layer_sizes:
            self.neuron_layers.append(LIFNeuron(device=device))
        
        # 初始化突触权重 - 使用ParameterList替代ModuleList
        self.weights = nn.ParameterList()
        for i in range(self.num_layers - 1):
            in_size = layer_sizes[i]
            out_size = layer_sizes[i+1]
            
            # 根据连接模式初始化权重
            if connection_mode == '1d':
                # 1D线性连接：1-2, 2-3, ...
                w = torch.zeros(in_size, out_size, device=device)
                for j in range(min(in_size, out_size)):
                    w[j, j] = 0.1  # 自连接
                    if j < out_size - 1:
                        w[j, j+1] = 0.1  # 前向连接
                    if j > 0:
                        w[j, j-1] = 0.1  # 反向连接
            elif connection_mode == '2d':
                # 2D网格连接：相邻神经元互相连接
                w = torch.zeros(in_size, out_size, device=device)
                # 假设输入和输出都是方阵
                in_dim = int(in_size ** 0.5)
                out_dim = int(out_size ** 0.5)
                
                for i_in in range(in_size):
                    x_in = i_in % in_dim
                    y_in = i_in // in_dim
                    
                    for i_out in range(out_size):
                        x_out = i_out % out_dim
                        y_out = i_out // out_dim
                        
                        # 曼哈顿距离<=1的神经元连接
                        if abs(x_in - x_out) + abs(y_in - y_out) <= 2:
                            w[i_in, i_out] = 0.1
            else:  # free mode
                # 无连接，权重初始化为0，后续手动指定
                w = torch.zeros(in_size, out_size, device=device)
            
            # 注册为可训练参数
            weight = nn.Parameter(w, requires_grad=True)
            self.weights.append(weight)
        
    def reset(self, batch_size):
        """重置所有神经元状态"""
        for i, neuron_layer in enumerate(self.neuron_layers):
            neuron_layer.reset(batch_size, self.layer_sizes[i])
    
    def forward(self, x_seq, gamma=0.9, sigma=0.1):
        """前向传播
        x_seq: 输入脉冲序列 [batch_size, time_steps, n_inputs]
        gamma: 时序痕迹衰减系数
        sigma: 一致性评分的标准差
        """
        batch_size, time_steps, n_inputs = x_seq.shape
        self.reset(batch_size)
        
        outputs = []
        gamma_list = []
        p_list = []
        
        for t in range(time_steps):
            x = x_seq[:, t, :]
            current_x = x
            
            for i in range(self.num_layers - 1):
                # 第i层神经元处理
                s, gamma_t = self.neuron_layers[i](current_x, self.weights[i], gamma, sigma)
                current_x = s
            
            # 输出层
            s_out, gamma_t_out = self.neuron_layers[-1](current_x, torch.eye(self.layer_sizes[-1], device=self.device), gamma, sigma)
            outputs.append(s_out)
            
            # 保存最后一层的gamma_t和p
            gamma_list.append(gamma_t_out)
            p_list.append(self.neuron_layers[-1].p)
        
        # 转换为张量
        outputs = torch.stack(outputs, dim=1)  # [batch_size, time_steps, n_outputs]
        gamma_t = torch.stack(gamma_list, dim=1)  # [batch_size, time_steps]
        p = torch.stack(p_list, dim=1)  # [batch_size, time_steps, n_outputs]
        
        return outputs, gamma_t, p
    
    def set_free_connections(self, layer_idx, connections):
        """手动设置free模式下的连接
        layer_idx: 要设置的层索引
        connections: 连接列表，每个元素为(from_neuron, to_neuron, weight)
        """
        if self.connection_mode != 'free':
            raise ValueError("This method is only available in free connection mode")
        
        for from_idx, to_idx, weight in connections:
            self.weights[layer_idx].data[from_idx, to_idx] = weight
    
    def save(self, filename):
        """保存模型参数"""
        # 确保save文件夹存在
        os.makedirs('save', exist_ok=True)
        # 构造完整路径
        path = os.path.join('save', filename)
        torch.save({
            'layer_sizes': self.layer_sizes,
            'connection_mode': self.connection_mode,
            'state_dict': self.state_dict()
        }, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path, device='cpu'):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            layer_sizes=checkpoint['layer_sizes'],
            connection_mode=checkpoint['connection_mode'],
            device=device
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model
