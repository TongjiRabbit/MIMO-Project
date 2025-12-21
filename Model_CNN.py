import sys
import os

# 1. 强制解决终端乱码
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import numpy as np
import glob

# ================= 配置区域 =================
# 训练超参数
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 400
# CNN 参数配置
INPUT_CHANNELS = 32  # 2(Real/Imag) * 4(Clusters) * 4(Bases)
HIDDEN_CHANNELS = 128 # 卷积核数量

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")
# ===========================================

# 1. 定义 CNN 神经网络模型
class ChannelNet(nn.Module):
    def __init__(self):
        super(ChannelNet, self).__init__()
        
        # 输入: (N, 32, 128, 10) -> (N, C, Freq, Users)
        
        self.features = nn.Sequential(
            # Layer 1: 32 -> 64
            nn.Conv2d(INPUT_CHANNELS, HIDDEN_CHANNELS, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(HIDDEN_CHANNELS),
            nn.ReLU(),
            
            # Layer 2: 64 -> 128
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(HIDDEN_CHANNELS * 2),
            nn.ReLU(),
            
            # Layer 3: 128 -> 64
            nn.Conv2d(HIDDEN_CHANNELS * 2, HIDDEN_CHANNELS, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(HIDDEN_CHANNELS),
            nn.ReLU(),
            
            # Layer 4 (Output Head): 64 -> 2 (Real, Imag)
            # 使用 1x1 卷积将特征映射回实部和虚部通道
            nn.Conv2d(HIDDEN_CHANNELS, 2, kernel_size=(1, 1)) 
        )

    def forward(self, x):
        # x shape: (N, 32, 128, 10)
        out = self.features(x) 
        # out shape: (N, 2, 128, 10) -> (Batch, Real/Imag, Freq, Users)
        
        # --- 输出适配 ---
        # 目标标签 Y 是 [Real_Flat, Imag_Flat]
        # P 的物理维度是 (Users, Freq) = (10, 128)
        # 所以我们需要调整维度顺序以匹配 Flatten 后的顺序
        
        # 1. 调整为 (N, 2, 10, 128) -> (Batch, Real/Imag, Users, Freq)
        out = out.permute(0, 1, 3, 2)
        
        # 2. 分离实部和虚部
        out_real = out[:, 0, :, :].contiguous() # (N, 10, 128)
        out_imag = out[:, 1, :, :].contiguous() # (N, 10, 128)
        
        # 3. 展平并拼接
        # Flatten (N, 10, 128) -> (N, 1280)
        out_real_flat = out_real.view(out_real.size(0), -1)
        out_imag_flat = out_imag.view(out_imag.size(0), -1)
        
        # Concat -> (N, 2560)
        return torch.cat([out_real_flat, out_imag_flat], dim=1)

def load_new_format_data():
    """
    加载数据并重塑为 CNN 格式 (N, 32, 128, 10)
    """
    X_list = []
    Y_list = []
    
    file_pattern = "TrainData_Batch_*.mat"
    files = glob.glob(file_pattern)
    
    if not files:
        raise ValueError(f"❌ 未找到任何匹配文件: {file_pattern}")
        
    print(f"\n>>> 发现 {len(files)} 个数据文件，开始加载...")
    
    count = 0
    for filename in files:
        try:
            mat_data = sio.loadmat(filename)
            if 'Batch_Buffer' not in mat_data: continue
            batch_buffer = mat_data['Batch_Buffer']
            num_samples = batch_buffer.shape[1]
            
            for i in range(num_samples):
                sample = batch_buffer[0, i]
                snr_val = float(sample['SNR_dB'].flat[0])
                
                # 过滤低 SNR
                if snr_val < 14: continue

                # --- 提取输入特征 X (R Matrix) ---
                # 原始维度: (128, 4, 10, 4) -> (Freq, Bases, Users, Clusters)
                r_real = sample['R_Real'] 
                r_imag = sample['R_Imag']
                
                # 定义 Reshape 函数
                def process_r(mat):
                    # 1. Transpose: (Freq, Bases, Users, Clusters) -> (Clusters, Bases, Freq, Users)
                    # axes=(3, 1, 0, 2)
                    mat_t = np.transpose(mat, (3, 1, 0, 2))
                    # 2. Reshape: 合并 Clusters(4) 和 Bases(4) -> 16
                    # 结果: (16, 128, 10)
                    return mat_t.reshape(-1, 128, 10)
                
                feat_real = process_r(r_real) # (16, 128, 10)
                feat_imag = process_r(r_imag) # (16, 128, 10)
                
                # Stack Real and Imag along channel dimension -> (32, 128, 10)
                x_cnn = np.concatenate((feat_real, feat_imag), axis=0)

                # --- 提取标签 Y (Precoders P) ---
                # 维度: (10, 128)
                p_real = sample['P_Real']
                p_imag = sample['P_Imag']
                # 保持 Flatten 输出以配合 Loss 计算 (N, 2560)
                y_vec = np.concatenate((p_real.flatten(), p_imag.flatten()))
                
                X_list.append(x_cnn)
                Y_list.append(y_vec)
                count += 1
                
            print(f"   [已加载] {filename}")
            
        except Exception as e:
            print(f"   [错误] 读取 {filename} 失败: {e}")

    if count == 0: raise ValueError("❌ 未成功加载任何样本数据！")

    X_all = np.array(X_list)
    Y_all = np.array(Y_list)
    
    print(f"\n✅ 数据加载完毕！")
    print(f"   输入维度 (CNN): {X_all.shape}") # (N, 32, 128, 10)
    print(f"   输出维度 (Flat): {Y_all.shape}")
    
    return torch.FloatTensor(X_all).to(device), torch.FloatTensor(Y_all).to(device)

def train():
    X_train, Y_train = load_new_format_data()
    
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 实例化 CNN 模型
    model = ChannelNet().to(device)
    
    # 定义优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 自定义损失函数 (保持不变)
    class CombinedLoss(nn.Module):
        def __init__(self, alpha=0.1): 
            super(CombinedLoss, self).__init__()
            self.mse = nn.MSELoss()
            self.cosine = nn.CosineSimilarity(dim=1)
            self.alpha = alpha

        def forward(self, pred, target):
            loss_mse = self.mse(pred, target)
            sim = self.cosine(pred, target)
            loss_cos = 1.0 - torch.mean(sim)
            return loss_mse + self.alpha * loss_cos
            
    criterion = CombinedLoss(alpha=0.1)
    
    print(f"\n=== 开始训练 CNN 模型 ===")
    model.train()
        
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x) # Output is flattened inside forward
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")
            
    print("训练完成！")
    
    # 保存模型 (CNN后缀)
    save_name = 'DFDCA_CNN_Model_OnlyTrainBatch2_only15and20db.pth'
    torch.save(model.state_dict(), save_name)
    print(f"✅ CNN 模型已保存至: {save_name}")

if __name__ == '__main__':
    train()