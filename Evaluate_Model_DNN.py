import sys
import os
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
from collections import defaultdict
import glob

# ================= 配置区域 =================
TEST_FILE_PATTERN = 'TestData_Batch_6.mat'  # 支持批量合并：自动查找所有匹配文件
MODEL_PATH = 'DFDCA_DNN_Colab_Model_TrainBatch1to5_-10to20gap5db_4096hidden_1000epoch.pth'                  # 训练好的模型
OUTPUT_FILE = 'DFDCA_Evaluation_Results_Colab_TrainBatch1to5_test6_4096hidden_1000epoch.mat'        # 导出的结果文件
HIDDEN_SIZE = 8192                                  # 必须与 Model.py 一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

# 1. 定义网络结构 (必须与训练代码完全一致)
class ChannelNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ChannelNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_SIZE),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(4096, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output_layer(x)

def calculate_metrics(p_pred_complex, p_true_complex):
    """ 计算单个样本的 MSE 和 余弦相似度 """
    # 展平为向量
    pred_flat = p_pred_complex.flatten()
    true_flat = p_true_complex.flatten()

    # 1. MSE 计算
    diff = pred_flat - true_flat
    mse = np.mean(np.abs(diff) ** 2)

    # 2. 余弦相似度计算 (考虑复数共轭)
    # Sim = |a . b*| / (|a| * |b|)
    dot_product = np.abs(np.vdot(pred_flat, true_flat)) 
    norm_pred = np.linalg.norm(pred_flat)
    norm_true = np.linalg.norm(true_flat)
    
    if norm_pred == 0 or norm_true == 0:
        similarity = 0.0
    else:
        similarity = dot_product / (norm_pred * norm_true)

    return mse, similarity

def main():
    print(f"=== DFDCA 模型验证与数据导出 ===")
    print(f"设备: {DEVICE}")

    # --- 1. 自动查找并合并所有匹配的测试数据 ---
    test_files = sorted(glob.glob(TEST_FILE_PATTERN))
    
    if not test_files:
        print(f"❌ 错误: 找不到任何匹配文件: {TEST_FILE_PATTERN}")
        return
    
    print(f"📂 发现 {len(test_files)} 个测试文件:")
    for f in test_files:
        print(f"   - {f}")
    
    # 合并所有 Batch_Buffer
    all_samples = []
    total_samples = 0
    
    for test_file in test_files:
        try:
            mat_data = sio.loadmat(test_file)
            test_buffer = mat_data['Batch_Buffer']
            num_batch_samples = test_buffer.shape[1]
            
            # 提取当前文件的所有样本
            for i in range(num_batch_samples):
                all_samples.append(test_buffer[0, i])
            
            total_samples += num_batch_samples
            print(f"   ✅ {test_file}: {num_batch_samples} 个样本")
        except Exception as e:
            print(f"   ❌ {test_file}: 读取失败 - {e}")
            continue
    
    if total_samples == 0:
        print(f"❌ 错误: 未成功加载任何样本")
        return
    
    print(f"\n✅ 成功合并 {total_samples} 个测试样本")

    # --- 2. 准备模型 ---
    # 读取第一个样本以确定输入输出维度
    sample_0 = all_samples[0]
    input_dim = sample_0['R_Real'].size + sample_0['R_Imag'].size
    output_dim = sample_0['P_Real'].size + sample_0['P_Imag'].size
    
    print(f"\n🔧 模型配置:")
    print(f"   输入维度: {input_dim}, 输出维度: {output_dim}")
    print(f"   隐层宽度: {HIDDEN_SIZE}")

    model = ChannelNet(input_dim, output_dim).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ 模型权重加载完毕")

    # --- 3. 推理与指标计算 ---
    # 使用字典存储按 SNR 分组的结果
    results_by_snr = defaultdict(lambda: {'mse': [], 'sim': [], 'power': []})

    print("\n开始推理计算...")
    with torch.no_grad():
        for i, sample in enumerate(all_samples):
            snr_val = float(sample['SNR_dB'].flat[0])

            # # 过滤逻辑：只测 15 和 20 dB
            # if snr_val < 14:
            #     continue



            # 获取数据
            r_real = sample['R_Real']
            r_imag = sample['R_Imag']
            p_real_true = sample['P_Real']
            p_imag_true = sample['P_Imag']
            
    
            # 预处理输入
            x_vec = np.concatenate((r_real.flatten(), r_imag.flatten()))
            x_tensor = torch.FloatTensor(x_vec).unsqueeze(0).to(DEVICE)

            #snr作为输入特征时采用
            # x_feat = np.concatenate((r_real.flatten(), r_imag.flatten()))
            # snr_norm = snr_val / 20.0
            # x_vec = np.append(x_feat, snr_norm)
            # x_tensor = torch.FloatTensor(x_vec).unsqueeze(0).to(DEVICE)

            # 模型预测
            y_pred = model(x_tensor).cpu().numpy().squeeze()

            # 后处理：还原复数
            mid = len(y_pred) // 2
            p_pred_complex = y_pred[:mid] + 1j * y_pred[mid:]
            p_true_complex = p_real_true.flatten() + 1j * p_imag_true.flatten()

            # 计算指标
            mse, sim = calculate_metrics(p_pred_complex, p_true_complex)
            
            # --- 新增：计算真实标签的功率 (模的平方) ---
            p_true_power = np.mean(np.abs(p_true_complex)**2)

            # 存入列表
            results_by_snr[snr_val]['mse'].append(mse)
            results_by_snr[snr_val]['sim'].append(sim)
            results_by_snr[snr_val]['power'].append(p_true_power) # <--- 新增这行

            if (i + 1) % 500 == 0:
                print(f"   已处理 {i + 1}/{total_samples} ...")

    # --- 4. 汇总数据 ---
    snr_list = sorted(results_by_snr.keys())
    avg_mse_list = []
    avg_sim_list = []
    
    # 这里也可以选择保存所有样本的原始数据，以便画箱线图等，
    # 但通常画性能曲线只需要平均值。这里我们两者都准备。
    raw_mse_data = [] # 这是一个 cell 类似的结构列表
    raw_sim_data = []

    print("\n=== 验证结果 (平均值) ===")
    print(f"{'SNR (dB)':<10} | {'MSE':<10} | {'Similarity':<10}| {'True Power':<10}")
    print("-" * 50)

    for snr in snr_list:
        mses = results_by_snr[snr]['mse']
        sims = results_by_snr[snr]['sim']
        pwrs = results_by_snr[snr]['power'] # <--- 获取功率列表
        
        avg_mse = np.mean(mses)
        avg_sim = np.mean(sims)
        avg_pwr = np.mean(pwrs)             # <--- 计算平均功率
        
        avg_mse_list.append(avg_mse)
        avg_sim_list.append(avg_sim)
        
        # 收集原始数据以便导出 (numpy array)
        raw_mse_data.append(np.array(mses))
        raw_sim_data.append(np.array(sims))

        print(f"{snr:<10} | {avg_mse:<10.5f} | {avg_sim:<10.5f}| {avg_pwr:<10.5f}")

    # --- 5. 导出到 .mat 文件 ---
    export_data = {
        'SNR_Axis': np.array(snr_list),
        'MSE_Curve': np.array(avg_mse_list),
        'Sim_Curve': np.array(avg_sim_list),
        # 如果需要在MATLAB里做更细致的分析（如CDF图），可以使用下面的原始数据
        # 注意：由于不同SNR样本数可能稍有不同，scipy保存这种非对齐数据通常用object array
        'Raw_MSE_Distribution': np.array(raw_mse_data, dtype=object),
        'Raw_Sim_Distribution': np.array(raw_sim_data, dtype=object)
    }

    sio.savemat(OUTPUT_FILE, export_data)
    print(f"\n✅ 数据已导出至: {OUTPUT_FILE}")
if __name__ == '__main__':
    main()