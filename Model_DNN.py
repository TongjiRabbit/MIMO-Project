import sys
import os
import argparse

# 1. 尝试解决终端乱码（在某些环境下可能不存在 reconfigure）
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import numpy as np
import glob

# ================= 配置区域 =================
# 训练超参数
# 默认超参数（可通过命令行覆盖）
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 1000
HIDDEN_SIZE = 4096

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")
# ===========================================

# 1. 定义神经网络模型 (保持不变，但输入维度会自动适配)
class ChannelNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=HIDDEN_SIZE):
        super(ChannelNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def load_new_format_data(file_pattern="TrainData_Batch_*.mat"):
    """
    适配 Generate_TrainingData_2.m 生成的 struct array 格式
    文件名格式: TrainData_Batch_{id}.mat
    """
    X_list = []
    Y_list = []
    
    # 查找所有匹配的文件 (Batch 1 到 5)
    files = glob.glob(file_pattern)
    
    if not files:
        raise ValueError(f"❌ 未找到任何匹配文件: {file_pattern}，请确保 .mat 文件在当前目录下。")
        
    print(f"\n>>> 发现 {len(files)} 个数据文件，开始加载...")
    
    count = 0
    for filename in files:
        try:
            mat_data = sio.loadmat(filename)
            # MATLAB 中保存的是 'Batch_Buffer'，它是一个结构体数组
            # 在 scipy.io 中，它通常是 shape 为 (1, N) 的 object ndarray
            if 'Batch_Buffer' not in mat_data:
                print(f"   [跳过] {filename} (不包含 Batch_Buffer)")
                continue
                
            batch_buffer = mat_data['Batch_Buffer']
            
            # batch_buffer 是二维数组 [[struct1, struct2, ...]]
            #我们需要遍历其中的每一个样本
            num_samples = batch_buffer.shape[1]
            
            for i in range(num_samples):
                sample = batch_buffer[0, i]
                
                # snr_val = float(sample['SNR_dB'].flat[0])
                
                # 如果 SNR 小于 14 (即排除 -10, 0, 5, 10)，则跳过
                # if snr_val < 14:
                #     continue

                # --- 提取输入特征 X (R Matrix) ---
                # sample['R_Real'] 出来通常是 [[array]]，需要取内容
                # 维度通常是: (128, 4, 10, 4)
                r_real = sample['R_Real'] 
                r_imag = sample['R_Imag']
                
                # --- 提取标签 Y (Precoders P) ---
                # 维度通常是: (10, 128)
                p_real = sample['P_Real']
                p_imag = sample['P_Imag']

                # --- 数据展平 (Flatten) ---
                # DNN 需要 1D 向量输入。我们将多维矩阵拉直。
                # 输入 X: 拼接实部虚部 -> Flatten
                x_vec = np.concatenate((r_real.flatten(), r_imag.flatten()))
                
                # 输出 Y: 拼接实部虚部 -> Flatten
                y_vec = np.concatenate((p_real.flatten(), p_imag.flatten()))
                
                X_list.append(x_vec)
                Y_list.append(y_vec)
                count += 1
                
            print(f"   [已加载] {filename}")
            
        except Exception as e:
            print(f"   [错误] 读取 {filename} 失败: {e}")

    if count == 0:
        raise ValueError("❌ 未成功加载任何样本数据！")

    # 转换为 Numpy 数组
    X_all = np.array(X_list)
    Y_all = np.array(Y_list)
    
    print(f"\n✅ 数据加载完毕！")
    print(f"   总样本数: {X_all.shape[0]}")
    print(f"   输入维度 (Flattened): {X_all.shape[1]}")
    print(f"   输出维度 (Flattened): {Y_all.shape[1]}")
    
    # 转为 PyTorch Tensor
    X_tensor = torch.FloatTensor(X_all).to(device)
    Y_tensor = torch.FloatTensor(Y_all).to(device)
    
    return X_tensor, Y_tensor

def train(args):
    # 使用新的加载函数（支持自定义文件匹配）
    X_train, Y_train = load_new_format_data(args.data_pattern)
    
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 动态获取输入输出维度
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    
    model = ChannelNet(input_dim, output_dim, hidden_size=args.hidden_size).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"\n=== 开始训练 DNN 模型 ===")
    model.train()
    class CombinedLoss(nn.Module):
        def __init__(self, alpha=0.1): # alpha 调节 MSE 和 Cosine 的比重
            super(CombinedLoss, self).__init__()
            self.mse = nn.MSELoss()
            self.cosine = nn.CosineSimilarity(dim=1)

        def forward(self, pred, target):
            # 1. MSE Loss
            loss_mse = self.mse(pred, target)
            
            # 2. Cosine Similarity Loss (目标是最大化相似度，即最小化 1 - Sim)
            # 需要把 pred 和 target 重新看作复数向量处理，或者直接对实数拼接向量做 Cosine
            # 这里直接对拼接后的高维向量做 Cosine 也是有效的，因为方向一致性在实数域拼接后依然保留
            sim = self.cosine(pred, target)
            loss_cos = 1.0 - torch.mean(sim)
        
            return loss_mse + alpha * loss_cos
        
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss (MSE): {avg_loss:.6f}")
            
    print("训练完成！")

    # 保存模型
    os.makedirs(args.save_path, exist_ok=True)
    if args.save_name:
        save_name = args.save_name
    else:
        save_name = f'DFDCA_DNN_Model_hidden{args.hidden_size}_{args.epochs}epoch.pth'

    save_full = os.path.join(args.save_path, save_name)
    torch.save(model.state_dict(), save_full)
    print(f"✅ 模型已保存至: {save_full}")
    print(f"   - Input Dim: {input_dim}")
    print(f"   - Output Dim: {output_dim}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train DNN model (Colab friendly)')
    parser.add_argument('--data-pattern', type=str, default='TrainData_Batch_*.mat',
                        help='glob pattern to find training .mat files')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--hidden-size', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--save-path', type=str, default='.')
    parser.add_argument('--save-name', type=str, default=None,
                        help='optional model filename')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # 将解析的超参写回全局默认（用于 DataLoader / 优化器 位置）
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    HIDDEN_SIZE = args.hidden_size

    # 打印设备信息
    print(f"当前运行设备: {device}")

    train(args)