import sys
import os
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
from collections import defaultdict

# ================= 配置区域 =================
MODEL_PATH = 'DFDCA_DNN_Model_TrainBatch1to5_-10to20db_gap5db_4096hidden_500epoch.pth'                  # 训练好的模型

DATA_DIR = r'D:\CodeSpace\CodeOfMMSE_xUser_re_OnlyTrainToTrain\Data'
TEST_FILES_NAME = [
'TrainData_Batch_1.mat',
'TrainData_Batch_2.mat',
]
TEST_FILES = [os.path.join(DATA_DIR, f) for f in TEST_FILES_NAME]

RESULT_DIR = r'D:\CodeSpace\CodeOfMMSE_xUser_re_OnlyTrainToTrain\Result_P'
OUTPUT_FILE_NAME = 'DFDCA_Evaluation_Results_TrainBatch1to5_test1to2_4096hidden_500epoch.mat'        # 导出的结果文件
OUTPUT_FILE = os.path.join(RESULT_DIR, OUTPUT_FILE_NAME)

HIDDEN_SIZE = 4096                                  # 必须与 Model.py 一致
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


def _extract_scalar(sample, field_name, default=None):
    if field_name not in sample.dtype.names:
        return default
    try:
        return float(np.array(sample[field_name]).flat[0])
    except Exception:
        return default


def _extract_user_indices(sample, usrnum=None):
    if 'User_Indices' not in sample.dtype.names:
        return None
    arr = np.array(sample['User_Indices']).astype(np.int64, copy=False).reshape(-1)
    if usrnum is not None and arr.size != usrnum:
        if arr.size > usrnum:
            arr = arr[:usrnum]
        else:
            arr = np.pad(arr, (0, usrnum - arr.size), mode='constant', constant_values=0)
    return arr


def normalize_p_pred_flat_ri(y_pred_flat_ri, usrnum, frenum, eps=1e-9):
    """对预测P做逐用户行功率归一化；输入/输出均为展平RI拼接向量。"""
    half = y_pred_flat_ri.size // 2
    p_real = y_pred_flat_ri[:half].reshape((usrnum, frenum), order='C')
    p_imag = y_pred_flat_ri[half:].reshape((usrnum, frenum), order='C')
    p_complex = p_real + 1j * p_imag

    for i in range(usrnum):
        current_p = p_complex[i, :]
        current_power = float(np.sum(np.abs(current_p) ** 2))
        if current_power > eps:
            p_complex[i, :] = current_p * np.sqrt(float(frenum) / current_power)

    p_real_norm = np.real(p_complex).reshape(-1, order='C')
    p_imag_norm = np.imag(p_complex).reshape(-1, order='C')
    return np.concatenate((p_real_norm, p_imag_norm))

def main():
    print(f"=== DFDCA 模型验证与数据导出 ===")
    print(f"设备: {DEVICE}")

    # --- 1. 手动指定并拼接多个数据源（按顺序拼接；确保元信息一一对应） ---
    test_files = list(TEST_FILES)

    if not test_files:
        print("❌ 错误: TEST_FILES 为空，请在脚本顶部手动填写待评估的 .mat 文件名列表。")
        return

    print(f"📂 待评估文件数: {len(test_files)}")
    for f in test_files:
        print(f"   - {f}")
    
    # 合并所有 Batch_Buffer
    all_samples = []
    total_samples = 0
    
    for test_file in test_files:
        try:
            if not os.path.exists(test_file):
                print(f"   ❌ {test_file}: 文件不存在")
                continue
            mat_data = sio.loadmat(test_file)
            if 'Batch_Buffer' not in mat_data:
                print(f"   ❌ {test_file}: 不包含 Batch_Buffer")
                continue
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

    # --- 3. 推理与指标计算 + 导出所需原始数据 ---
    # 使用字典存储按 SNR 分组的结果
    results_by_snr = defaultdict(lambda: {'mse': [], 'sim': [], 'power': []})

    # 逐样本导出（与 all_samples 顺序一一对应）
    p_pred_flat_ri_list = []
    p_true_flat_ri_list = []
    snr_db_list = []
    noise_power_list = []
    group_id_list = []
    user_indices_list = []

    print("\n开始推理计算...")
    with torch.no_grad():
        for i, sample in enumerate(all_samples):
            snr_val = _extract_scalar(sample, 'SNR_dB', default=np.nan)

            # # 过滤逻辑：只测 15 和 20 dB
            # if snr_val < 14:
            #     continue



            # 获取数据
            r_real = sample['R_Real']
            r_imag = sample['R_Imag']
            p_real_true = sample['P_Real']
            p_imag_true = sample['P_Imag']

            usrnum = int(p_real_true.shape[0]) if hasattr(p_real_true, 'shape') and np.ndim(p_real_true) >= 2 else 10
            total_p = int(np.size(p_real_true))
            if hasattr(p_real_true, 'shape') and np.ndim(p_real_true) >= 2:
                frenum = int(p_real_true.shape[1])
            else:
                if usrnum <= 0 or total_p % usrnum != 0:
                    raise ValueError(f"无法从 P_Real 推断维度: size={total_p}, usrnum={usrnum}")
                frenum = int(total_p // usrnum)
            
    
            # 预处理输入
            x_vec = np.concatenate((r_real.flatten(), r_imag.flatten()))
            x_tensor = torch.FloatTensor(x_vec).unsqueeze(0).to(DEVICE)

            #snr作为输入特征时采用
            # x_feat = np.concatenate((r_real.flatten(), r_imag.flatten()))
            # snr_norm = snr_val / 20.0
            # x_vec = np.append(x_feat, snr_norm)
            # x_tensor = torch.FloatTensor(x_vec).unsqueeze(0).to(DEVICE)

            # 模型预测
            y_pred = model(x_tensor).cpu().numpy().squeeze().astype(np.float64, copy=False)

            # --- 预测P功率归一化（逐用户行） ---
            y_pred_norm = normalize_p_pred_flat_ri(y_pred, usrnum=usrnum, frenum=frenum)

            # 后处理：用于指标计算的复数向量（展平）
            mid = len(y_pred_norm) // 2
            p_pred_complex = y_pred_norm[:mid] + 1j * y_pred_norm[mid:]
            p_true_complex = p_real_true.flatten(order='C') + 1j * p_imag_true.flatten(order='C')

            # 计算指标
            mse, sim = calculate_metrics(p_pred_complex, p_true_complex)
            
            # --- 新增：计算真实标签的功率 (模的平方) ---
            p_true_power = np.mean(np.abs(p_true_complex)**2)

            # --- 收集导出数据（保持展平，不复原为矩阵） ---
            p_pred_flat_ri_list.append(y_pred_norm)
            p_true_flat_ri_list.append(
                np.concatenate((p_real_true.flatten(order='C'), p_imag_true.flatten(order='C'))).astype(np.float64, copy=False)
            )
            snr_db_list.append(snr_val)
            noise_power_list.append(_extract_scalar(sample, 'Noise_Power', default=np.nan))
            group_id_list.append(_extract_scalar(sample, 'Group_ID', default=np.nan))
            user_idx = _extract_user_indices(sample, usrnum=usrnum)
            user_indices_list.append(user_idx if user_idx is not None else np.zeros((usrnum,), dtype=np.int64))

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
    p_pred_flat_ri = np.stack(p_pred_flat_ri_list, axis=0) if p_pred_flat_ri_list else np.zeros((0, output_dim))
    p_true_flat_ri = np.stack(p_true_flat_ri_list, axis=0) if p_true_flat_ri_list else np.zeros((0, output_dim))
    user_indices = np.stack(user_indices_list, axis=0) if user_indices_list else np.zeros((0, 0), dtype=np.int64)

    export_data = {
        # 基础配置
        'Model_Path': MODEL_PATH,
        'Input_Dim': np.array([input_dim], dtype=np.int64),
        'Output_Dim': np.array([output_dim], dtype=np.int64),
        # 逐样本导出（顺序与 all_samples 完全一致）
        'P_Pred_Flat_RI': p_pred_flat_ri,
        'P_True_Flat_RI': p_true_flat_ri,
        'SNR_dB': np.array(snr_db_list, dtype=np.float64),
        'Noise_Power': np.array(noise_power_list, dtype=np.float64),
        'Group_ID': np.array(group_id_list, dtype=np.float64),
        'User_Indices': user_indices,
        # 按SNR统计结果（原有）
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