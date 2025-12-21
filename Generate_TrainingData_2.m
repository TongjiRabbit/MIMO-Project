%% Generate_TrainingData_Parallel.m
% 功能：并行生成多SNR下的 DFDCA 训练数据
% 优化：使用 parfor 多核加速，速度提升 N 倍 (N=核心数)

clc; clear; close all;

%% 1. 全局配置
disp('=== DFDCA 训练数据生成 (多核并行版) ===');

% 检测并开启并行池
currentPool = gcp('nocreate');
if isempty(currentPool)
    parpool; % 自动开启最大核心数
end

% 加载信道源数据
dataFileName = 'wcnc_18_40_129_102slot_nloss.mat';
if ~exist(dataFileName, 'file'), error('找不到信道文件！'); end
load(dataFileName);

% 截取训练用 Slot (前80个)
Training_Slot_Range = 1:80;
H_source_total = H_wcnc(:, :, :, Training_Slot_Range); 
[Antnum, Total_Pool_Size, Frenum, Slotnum] = size(H_source_total);

% 参数配置
Target_Usrnum = 10;
K_groups = 4;
iter_max = 5;  % 5次迭代通常足够，减少计算量
SNR_List = [-10, 0, 5, 10, 15, 20]; % 覆盖低信噪比到高信噪比

Num_Groups = 2000; % 建议 2000 组，产生 2000 * 6 = 12000 个样本
User_Groups = zeros(Num_Groups, Target_Usrnum);
rng('shuffle'); 
for g = 1:Num_Groups
    User_Groups(g, :) = sort(randperm(Total_Pool_Size, Target_Usrnum));
end

% DFT 矩阵准备
WN = exp(-1i*2*pi/Frenum); 
k_initial = 0:Frenum-1; 
if Target_Usrnum == 10, k0 = 16; else, k0 = 16; end % 根据实际修改

disp(['准备生成 ', num2str(Num_Groups), ' 组用户数据，每组包含 ', num2str(length(SNR_List)), ' 个SNR场景...']);
tic;

%% 2. 并行计算核心 (Parfor)
% 预分配 Cell 数组存储结果，避免 parfor 竞争
Results_Cell = cell(1, Num_Groups);

parfor g_idx = 1:Num_Groups
    current_users = User_Groups(g_idx, :);
    
    % --- A. 信道提取与归一化 ---
    H_intial2_tmp = zeros(Antnum, Target_Usrnum, Frenum, Slotnum);
    for u_local = 1:Target_Usrnum
        u_global = current_users(u_local);
        H_intial2_tmp(:, u_local, :, :) = H_source_total(:, u_global, :, :);
    end
    
    % 功率归一化
    for usr = 1:Target_Usrnum
        for ant = 1:Antnum
            H_vec = squeeze(H_intial2_tmp(ant, usr, :, :));
            H_pwr = sqrt(trace(H_vec' * H_vec) / Frenum / Slotnum);
            if H_pwr > 0, H_intial2_tmp(ant, usr, :, :) = H_vec ./ H_pwr; end
        end
    end
    
    % 施加相移
    H_intial2 = zeros(Antnum, Target_Usrnum, Frenum, Slotnum);
    for ant = 1:Antnum
        for usr = 1:Target_Usrnum
            D2 = WN .^ (k_initial * (usr-1) * k0);
            H_intial2(ant, usr, :, :) = transpose(D2) .* squeeze(H_intial2_tmp(ant, usr, :, :));
        end
    end
    
    % --- B. 统计特征 R ---
    Antnum_per_group = Antnum / K_groups;
    R = zeros(Frenum, Frenum, Target_Usrnum, K_groups);
    R_central_pro = zeros(Frenum, Frenum, Target_Usrnum, K_groups);
    R_equal = zeros(Frenum, 4, Target_Usrnum, K_groups); 
    
    for k_grp = 1:K_groups
        ant_start = (k_grp - 1) * Antnum_per_group + 1; 
        ant_end = k_grp * Antnum_per_group;
        for usridx = 1:Target_Usrnum
            for antidx = ant_start:ant_end
                for slotidx = 1:Slotnum
                    H_tmp = squeeze(H_intial2(antidx, usridx, :, slotidx));
                    R(:,:,usridx,k_grp) = R(:,:,usridx,k_grp) + H_tmp * H_tmp';
                end
            end
        end
    end
    R = R ./ (Antnum_per_group * Slotnum);
    
    k = 4; k_central_pro = 6;
    U = zeros(Frenum, k, Target_Usrnum, K_groups);
    for usridx = 1:Target_Usrnum
        for k_grp = 1:K_groups
            [U_tmp, S_tmp, ~] = svd(R(:,:,usridx,k_grp)); 
            U(:,:,usridx,k_grp) = U_tmp(:, 1:k);
            R_equal(:,:,usridx,k_grp) = U_tmp(:, 1:k)*sqrt(S_tmp(1:k,1:k)); % DNN输入特征
            R_central_pro(:,:,usridx,k_grp) = U_tmp(:,1:k_central_pro)*S_tmp(1:k_central_pro,1:k_central_pro)*U_tmp(:,1:k_central_pro)';
        end
    end
    
    c_2 = zeros(k, Target_Usrnum, K_groups);
    for usridx = 1:Target_Usrnum
        for k_grp = 1:K_groups
            tmpU1 = squeeze(U(:,:,usridx,k_grp)); 
            c_tmp = zeros(k, 1);
            ant_start = (k_grp-1)*Antnum_per_group+1; ant_end = k_grp*Antnum_per_group;
            for antidx = ant_start:ant_end
                for slotidx = 1:Slotnum
                    H_tmp = squeeze(H_intial2(antidx, usridx, :, slotidx)); 
                    c_tmp_tmp = tmpU1' * H_tmp;
                    for k_idx = 1:k, c_tmp(k_idx) = c_tmp(k_idx) + c_tmp_tmp(k_idx) * c_tmp_tmp(k_idx)'; end
                end
            end
            for k_idx = 1:k, c_2(k_idx,usridx,k_grp) = c_tmp(k_idx)/(Antnum_per_group*Slotnum); end
        end
    end

    % --- C. 多 SNR 循环 (MMSE求解) ---
    Group_Samples = []; % 临时存该组的6个样本
    
    for s_idx = 1:length(SNR_List)
        snr_db = SNR_List(s_idx);
        
        % 估算噪声
        noise_power_tmp = 0; sample_slots = 5:10:75; 
        for tidx = sample_slots
            H_tidx = squeeze(H_intial2(:, 1:Target_Usrnum, :, tidx));
            signal_power = sum(abs(H_tidx(:)).^2) / numel(H_tidx);
            sol_snr = 10^(snr_db/10);
            noise_power_tmp = noise_power_tmp + sqrt(signal_power/sol_snr);
        end
        noise_power = noise_power_tmp / length(sample_slots) * 0.9;
        
        P = ones(Target_Usrnum, Frenum);
        W = ones(Target_Usrnum, k, Frenum, K_groups) / sqrt(Frenum);
        
        % MMSE 迭代
        for idx_iter = 1:iter_max
            % Receiver
            for k_grp = 1:K_groups
                SSH_total_k = zeros(Frenum, Frenum);
                for i = 1:Target_Usrnum
                    tmpP1 = squeeze(P(i,:)); tmpP2 = diag(tmpP1); 
                    tmpR = squeeze(R(:,:,i,k_grp)); 
                    SSH_total_k = SSH_total_k + tmpP2 * tmpR * tmpP2'; 
                end
                Sc_k = zeros(Frenum, k, Target_Usrnum);
                for i = 1:Target_Usrnum
                    for j = 1:k
                        X_k = squeeze(U(:,j,i,k_grp)) .* transpose(P(i,:)); 
                        Sc_k(:,j,i) = X_k .* c_2(j,i,k_grp); 
                    end
                end
                matrix_inv = pinv(SSH_total_k + noise_power^2*eye(Frenum) + 1e-9*eye(Frenum));
                for i = 1:Target_Usrnum
                    for j = 1:k, W(i,j,:,k_grp) = matrix_inv * Sc_k(:,j,i); end
                end
            end
            % Transmitter
            if idx_iter < iter_max
                for i_target = 1:Target_Usrnum
                    A_i = zeros(Frenum, Frenum);
                    for m_sum = 1:Target_Usrnum
                        for j_sum = 1:k
                            for k_sum = 1:K_groups
                                W_vec = squeeze(W(m_sum, j_sum, :, k_sum));
                                R_approx_conj_tmp = conj(squeeze(R_central_pro(:,:,i_target,k_sum)));
                                A_i = A_i + diag(W_vec) * R_approx_conj_tmp * diag(W_vec)';
                            end
                        end
                    end
                    b_i = zeros(Frenum, 1);
                    for j_sum = 1:k
                        for k_sum = 1:K_groups
                            U_vec = squeeze(U(:, j_sum, i_target, k_sum));
                            W_vec = squeeze(W(i_target, j_sum, :, k_sum));
                            c2_val = c_2(j_sum, i_target, k_sum);
                            b_i = b_i + diag(U_vec)' * (W_vec) * c2_val;100
                        end
                    end
                    tmp_ip = A_i; tmp_sc = b_i;
                    p_unc = pinv(tmp_ip + 1e-9*eye(Frenum)) * tmp_sc;
                    if (p_unc' * p_unc) > Frenum
                        try lamda_p = lamdacal(tmp_ip, tmp_sc, Frenum); catch, lamda_p = 0.5; end
                    else, lamda_p = 0; end
                    tmp_p = pinv(tmp_ip + lamda_p*eye(Frenum) + 1e-9*eye(Frenum)) * tmp_sc;
                    p_pwr = tmp_p' * tmp_p;
                    if p_pwr > 1e-9, P(i_target,:) = tmp_p * sqrt(Frenum/p_pwr); end
                end
            end
        end 
        
        % 保存结果
        SampleStruct = struct();
        SampleStruct.R_Real = real(R_equal);
        SampleStruct.R_Imag = imag(R_equal);
        SampleStruct.P_Real = real(P);
        SampleStruct.P_Imag = imag(P);
        SampleStruct.User_Indices = current_users;
        SampleStruct.Noise_Power = noise_power;
        SampleStruct.SNR_dB = snr_db;
        SampleStruct.Group_ID = g_idx;
        
        if isempty(Group_Samples), Group_Samples = SampleStruct;
        else, Group_Samples(end+1) = SampleStruct; end
    end
    
    Results_Cell{g_idx} = Group_Samples;
    if mod(g_idx, 50) == 0, fprintf('Worker 已完成 Group %d\n', g_idx); end
end

%% 3. 数据组装与保存
disp('计算完成，正在组装数据...');
All_Data = [Results_Cell{:}]; % Cell 转 Struct Array

% 每 2000 个样本存一个文件 (防止文件过大)
Samples_Per_File = 2000;
Total_Samples = length(All_Data);
Num_Files = ceil(Total_Samples / Samples_Per_File);

for b = 1:Num_Files
    idx_start = (b-1)*Samples_Per_File + 1;
    idx_end = min(b*Samples_Per_File, Total_Samples);
    
    Batch_Buffer = All_Data(idx_start:idx_end);
    
    save_filename = sprintf('TrainData_MultiSNR_Batch_%d.mat', b);
    fprintf('正在保存 %s (样本 %d-%d)...\n', save_filename, idx_start, idx_end);
    save(save_filename, 'Batch_Buffer');
end

fprintf('训练数据生成完毕！总耗时: %.2f 分钟\n', toc/60);