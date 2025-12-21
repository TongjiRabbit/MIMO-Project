import sys
import os
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
from collections import defaultdict

# ================= 配置区域 =================
TEST_FILE = 'TrainData_Batch_2.mat' 
# 注意：这里加载的是 CNN 模型
MODEL_PATH = 'DFDCA_CNN_Model_OnlyTrainBatch2_only15and20db.pth'                  
OUTPUT_FILE = 'DFDCA_Evaluation_CNN_Results_OnlyTrainBatch_2test2_only15and20.mat'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CNN 配置
INPUT_CHANNELS = 32
HIDDEN_CHANNELS = 128
# ===========================================

# 1. 定义网络结构 (必须与训练代码 ChannelNet 完全一致)
class ChannelNet(nn.Module):
    def __init__(self):
        super(ChannelNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, HIDDEN_CHANNELS, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(HIDDEN_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(HIDDEN_CHANNELS * 2),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_CHANNELS * 2, HIDDEN_CHANNELS, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(HIDDEN_CHANNELS),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_CHANNELS, 2, kernel_size=(1, 1)) 
        )

    def forward(self, x):
        out = self.features(x) 
        out = out.permute(0, 1, 3, 2) # (N, 2, 10, 128)
        out_real = out[:, 0, :, :].contiguous()
        out_imag = out[:, 1, :, :].contiguous()
        out_real_flat = out_real.view(out_real.size(0), -1)
        out_imag_flat = out_imag.view(out_imag.size(0), -1)
        return torch.cat([out_real_flat, out_imag_flat], dim=1)

def calculate_metrics(p_pred_complex, p_true_complex):
    """ 计算单个样本的 MSE 和 余弦相似度 """
    pred_flat = p_pred_complex.flatten()
    true_flat = p_true_complex.flatten()

    diff = pred_flat - true_flat
    mse = np.mean(np.abs(diff) ** 2)

    dot_product = np.abs(np.vdot(pred_flat, true_flat)) 
    norm_pred = np.linalg.norm(pred_flat)
    norm_true = np.linalg.norm(true_flat)
    
    if norm_pred == 0 or norm_true == 0:
        similarity = 0.0
    else:
        similarity = dot_product / (norm_pred * norm_true)

    return mse, similarity

def main():
    print(f"=== DFDCA CNN 模型验证与数据导出 ===")
    print(f"设备: {DEVICE}")

    # --- 1. 加载测试数据 ---
    if not os.path.exists(TEST_FILE):
        print(f"❌ 错误: 找不到测试文件 {TEST_FILE}")
        return

    try:
        print(f"正在加载 {TEST_FILE} ...")
        mat_data = sio.loadmat(TEST_FILE)
        test_buffer = mat_data['Batch_Buffer'] 
    except Exception as e:
        print(f"❌ 读取 MAT 文件失败: {e}")
        return

    num_samples = test_buffer.shape[1]
    print(f"✅ 成功加载 {num_samples} 个测试样本")

    # --- 2. 准备模型 ---
    model = ChannelNet().to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ CNN 模型权重加载完毕")

    # --- 3. 推理与指标计算 ---
    results_by_snr = defaultdict(lambda: {'mse': [], 'sim': [], 'power': []})

    print("开始推理计算...")
    with torch.no_grad():
        for i in range(num_samples):
            sample = test_buffer[0, i]
            
            snr_val = float(sample['SNR_dB'].flat[0])

            # 过滤逻辑
            if snr_val < 14:
                continue

            # 获取数据
            r_real = sample['R_Real'] # (128, 4, 10, 4)
            r_imag = sample['R_Imag']
            p_real_true = sample['P_Real']
            p_imag_true = sample['P_Imag']
            
            # --- 预处理输入 (与训练代码严格一致) ---
            def process_r(mat):
                # Transpose (Freq, Bases, Users, Clusters) -> (Clusters, Bases, Freq, Users)
                mat_t = np.transpose(mat, (3, 1, 0, 2))
                # Reshape -> (16, 128, 10)
                return mat_t.reshape(-1, 128, 10)

            feat_real = process_r(r_real)
            feat_imag = process_r(r_imag)
            
            # (32, 128, 10)
            x_vec = np.concatenate((feat_real, feat_imag), axis=0)
            
            # 增加 Batch 维 -> (1, 32, 128, 10)
            x_tensor = torch.FloatTensor(x_vec).unsqueeze(0).to(DEVICE)

            # 模型预测 (输出是 Flatten 后的)
            y_pred = model(x_tensor).cpu().numpy().squeeze()

            # 后处理：还原复数
            # 注意：Model 输出顺序是 [Real_Flat, Imag_Flat]
            mid = len(y_pred) // 2
            p_pred_complex = y_pred[:mid] + 1j * y_pred[mid:]
            p_true_complex = p_real_true.flatten() + 1j * p_imag_true.flatten()

            # 计算指标
            mse, sim = calculate_metrics(p_pred_complex, p_true_complex)
            p_true_power = np.mean(np.abs(p_true_complex)**2)

            results_by_snr[snr_val]['mse'].append(mse)
            results_by_snr[snr_val]['sim'].append(sim)
            results_by_snr[snr_val]['power'].append(p_true_power)

            if (i + 1) % 500 == 0:
                print(f"   已处理 {i + 1}/{num_samples} ...")

    # --- 4. 汇总数据 ---
    snr_list = sorted(results_by_snr.keys())
    avg_mse_list = []
    avg_sim_list = []
    
    raw_mse_data = [] 
    raw_sim_data = []

    print("\n=== 验证结果 (平均值) ===")
    print(f"{'SNR (dB)':<10} | {'MSE':<10} | {'Similarity':<10}| {'True Power':<10}")
    print("-" * 50)

    for snr in snr_list:
        mses = results_by_snr[snr]['mse']
        sims = results_by_snr[snr]['sim']
        pwrs = results_by_snr[snr]['power']
        
        avg_mse = np.mean(mses)
        avg_sim = np.mean(sims)
        avg_pwr = np.mean(pwrs)
        
        avg_mse_list.append(avg_mse)
        avg_sim_list.append(avg_sim)
        
        raw_mse_data.append(np.array(mses))
        raw_sim_data.append(np.array(sims))

        print(f"{snr:<10} | {avg_mse:<10.5f} | {avg_sim:<10.5f}| {avg_pwr:<10.5f}")

    # --- 5. 导出到 .mat 文件 ---
    export_data = {
        'SNR_Axis': np.array(snr_list),
        'MSE_Curve': np.array(avg_mse_list),
        'Sim_Curve': np.array(avg_sim_list),
        'Raw_MSE_Distribution': np.array(raw_mse_data, dtype=object),
        'Raw_Sim_Distribution': np.array(raw_sim_data, dtype=object)
    }

    sio.savemat(OUTPUT_FILE, export_data)
    print(f"\n✅ 数据已导出至: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()