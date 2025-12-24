import sys
import os
import torch
import torch.nn as nn
import scipy.io as sio
import numpy as np
from collections import defaultdict

# ================= 配置区域 =================
MODEL_PATH = 'DFDCA_CNN_Model_TrainBatch1to5_-10to20db_gap5db_4096hidden_500epoch.pth'                  # 训练好的模型

DATA_DIR = r'D:\CodeSpace\CodeOfMMSE_xUser_re_OnlyTrainToTrain\Data'
TEST_FILES_NAME = [
'TrainData_Batch_1.mat',
'TrainData_Batch_2.mat',
]
TEST_FILES = [os.path.join(DATA_DIR, f) for f in TEST_FILES_NAME]

RESULT_DIR = r'D:\CodeSpace\CodeOfMMSE_xUser_re_OnlyTrainToTrain\Result_P'
OUTPUT_FILE_NAME = 'DFDCA_Evaluation_Results_TrainBatch1to5_test1to2_4096hidden_500epoch.mat'        # 导出的结果文件
OUTPUT_FILE = os.path.join(RESULT_DIR, OUTPUT_FILE_NAME)
# CNN 配置
INPUT_CHANNELS = 32
HIDDEN_CHANNELS = 256
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
    print(f"=== DFDCA CNN 模型验证与数据导出 ===")
    print(f"设备: {DEVICE}")

    # --- 1. 手动指定并拼接多个数据源（按顺序拼接；确保元信息一一对应） ---
    test_files = list(TEST_FILES)

    if not test_files:
        print("❌ 错误: TEST_FILES 为空，请在脚本顶部手动填写待评估的 .mat 文件名列表。")
        return

    print(f"📂 待评估文件数: {len(test_files)}")
    for f in test_files:
        print(f"   - {f}")

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

            for i in range(num_batch_samples):
                all_samples.append(test_buffer[0, i])

            total_samples += num_batch_samples
            print(f"   ✅ {test_file}: {num_batch_samples} 个样本")
        except Exception as e:
            print(f"   ❌ {test_file}: 读取失败 - {e}")
            continue

    if total_samples == 0:
        print("❌ 错误: 未成功加载任何样本")
        return

    print(f"\n✅ 成功合并 {total_samples} 个测试样本")

    # --- 2. 准备模型 ---
    model = ChannelNet().to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ CNN 模型权重加载完毕")

    # --- 3. 推理与指标计算 + 导出所需原始数据 ---
    results_by_snr = defaultdict(lambda: {'mse': [], 'sim': [], 'power': []})

    # 逐样本导出（与 all_samples 顺序一一对应）
    p_pred_flat_ri_list = []
    p_true_flat_ri_list = []
    snr_db_list = []
    noise_power_list = []
    group_id_list = []
    user_indices_list = []

    print("开始推理计算...")
    with torch.no_grad():
        for i, sample in enumerate(all_samples):
            snr_val = _extract_scalar(sample, 'SNR_dB', default=np.nan)

            # 过滤逻辑
            #if snr_val < 14:
               # continue

            # 获取数据
            r_real = sample['R_Real'] # (128, 4, 10, 4)
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
            y_pred = model(x_tensor).cpu().numpy().squeeze().astype(np.float64, copy=False)

            # --- 预测P功率归一化（逐用户行） ---
            y_pred_norm = normalize_p_pred_flat_ri(y_pred, usrnum=usrnum, frenum=frenum)

            # 后处理：还原复数
            # 注意：Model 输出顺序是 [Real_Flat, Imag_Flat]
            mid = len(y_pred_norm) // 2
            p_pred_complex = y_pred_norm[:mid] + 1j * y_pred_norm[mid:]
            p_true_complex = p_real_true.flatten(order='C') + 1j * p_imag_true.flatten(order='C')

            # 计算指标
            mse, sim = calculate_metrics(p_pred_complex, p_true_complex)
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

            results_by_snr[snr_val]['mse'].append(mse)
            results_by_snr[snr_val]['sim'].append(sim)
            results_by_snr[snr_val]['power'].append(p_true_power)

            if (i + 1) % 500 == 0:
                print(f"   已处理 {i + 1}/{total_samples} ...")

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
    # 逐样本导出
    if p_pred_flat_ri_list:
        p_pred_flat_ri = np.stack(p_pred_flat_ri_list, axis=0)
        p_true_flat_ri = np.stack(p_true_flat_ri_list, axis=0)
        user_indices = np.stack(user_indices_list, axis=0)
    else:
        p_pred_flat_ri = np.zeros((0, 0), dtype=np.float64)
        p_true_flat_ri = np.zeros((0, 0), dtype=np.float64)
        user_indices = np.zeros((0, 0), dtype=np.int64)

    export_data = {
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
        'Raw_MSE_Distribution': np.array(raw_mse_data, dtype=object),
        'Raw_Sim_Distribution': np.array(raw_sim_data, dtype=object)
    }

    sio.savemat(OUTPUT_FILE, export_data)
    print(f"\n✅ 数据已导出至: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()