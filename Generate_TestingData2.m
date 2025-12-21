%% Generate_TestingData_Parallel_V2.m
% 功能：带实时进度显示的并行测试集生成
% 核心改进：引入 DataQueue 实现 parfor 实时打印

clc; clear; close all;

disp('=== DFDCA 测试数据生成 (带进度显示版) ===');

% 1. 开启并行池
pool_size = 8; 
currentPool = gcp('nocreate');
if isempty(currentPool)
    parpool(pool_size);
elseif currentPool.NumWorkers > pool_size
    delete(currentPool);
    parpool(pool_size);
end

% 2. 加载数据
fprintf('正在加载信道数据...\n');
load('wcnc_18_40_129_102slot_nloss.mat');

% 锁定后 20 个 Slot (测试集)
Testing_Slot_Range = 81:100;
H_source_total = H_wcnc(:, :, :, Testing_Slot_Range); 
[Antnum, Total_Pool_Size, Frenum, Slotnum] = size(H_source_total);

% 参数配置
Target_Usrnum = 10;
K_groups = 4;
iter_max = 5; % 测试集迭代5次
SNR_List = -10:2:20;
Num_Groups = 50; 

% 随机用户组
User_Groups = zeros(Num_Groups, Target_Usrnum);
rng(42); 
for g = 1:Num_Groups
    User_Groups(g, :) = sort(randperm(Total_Pool_Size, Target_Usrnum));
end

% DFT 参数
WN = exp(-1i*2*pi/Frenum); k_initial = 0:Frenum-1; k0 = 16;

% === 3. 设置并行进度条 ===
D = parallel.pool.DataQueue;
afterEach(D, @updateProgress); % 设置回调函数

fprintf('开始生成 %d 组测试数据...\n', Num_Groups);
fprintf('进度: 0%%');

Results_Cell = cell(1, Num_Groups);
start_tic = tic;

parfor g_idx = 1:Num_Groups
    % --- 数据计算逻辑 (完全复用之前代码) ---
    current_users = User_Groups(g_idx, :);
    H_intial2_tmp = zeros(Antnum, Target_Usrnum, Frenum, Slotnum);
    
    % 信道提取
    for u_local = 1:Target_Usrnum
        u_global = current_users(u_local);
        H_intial2_tmp(:, u_local, :, :) = H_source_total(:, u_global, :, :);
    end
    
    % 归一化
    for usr = 1:Target_Usrnum
        for ant = 1:Antnum
            H_vec = squeeze(H_intial2_tmp(ant, usr, :, :));
            H_pwr = sqrt(trace(H_vec' * H_vec) / Frenum / Slotnum);
            if H_pwr > 0, H_intial2_tmp(ant, usr, :, :) = H_vec ./ H_pwr; end
        end
    end
    
    % DFT 相移
    H_intial2 = zeros(Antnum, Target_Usrnum, Frenum, Slotnum);
    for ant = 1:Antnum
        for usr = 1:Target_Usrnum
            D2 = WN .^ (k_initial * (usr-1) * k0);
            H_intial2(ant, usr, :, :) = transpose(D2) .* squeeze(H_intial2_tmp(ant, usr, :, :));
        end
    end
    
    % 计算 R
    Antnum_per_group = Antnum / K_groups;
    R = zeros(Frenum, Frenum, Target_Usrnum, K_groups);
    R_central_pro = zeros(Frenum, Frenum, Target_Usrnum, K_groups);
    R_equal = zeros(Frenum, 4, Target_Usrnum, K_groups);
    
    for k_grp = 1:K_groups
        ant_start = (k_grp - 1) * Antnum_per_group + 1; ant_end = k_grp * Antnum_per_group;
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
    
    % SVD
    k = 4; k_central_pro = 6;
    U = zeros(Frenum, k, Target_Usrnum, K_groups);
    for usridx = 1:Target_Usrnum
        for k_grp = 1:K_groups
            [U_tmp, S_tmp, ~] = svd(R(:,:,usridx,k_grp)); 
            U(:,:,usridx,k_grp) = U_tmp(:, 1:k);
            R_equal(:,:,usridx,k_grp) = U_tmp(:, 1:k)*sqrt(S_tmp(1:k,1:k));
            R_central_pro(:,:,usridx,k_grp) = U_tmp(:,1:k_central_pro)*S_tmp(1:k_central_pro,1:k_central_pro)*U_tmp(:,1:k_central_pro)';
        end
    end
    
    % 计算 c_2
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
    
    Group_Samples = [];
    
    % 多 SNR 循环
    for s_idx = 1:length(SNR_List)
        snr_db = SNR_List(s_idx);
        noise_power_tmp = 0; sample_slots = 5:10:min(Slotnum, 75); 
        for tidx = sample_slots
            H_tidx = squeeze(H_intial2(:, 1:Target_Usrnum, :, tidx));
            signal_power = sum(abs(H_tidx(:)).^2) / numel(H_tidx);
            sol_snr = 10^(snr_db/10);
            noise_power_tmp = noise_power_tmp + sqrt(signal_power/sol_snr);
        end
        noise_power = noise_power_tmp / length(sample_slots) * 0.9;
        P = ones(Target_Usrnum, Frenum);
        W = ones(Target_Usrnum, k, Frenum, K_groups) / sqrt(Frenum);
        for idx_iter = 1:iter_max
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
                            b_i = b_i + diag(U_vec)' * (W_vec) * c2_val;
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
        SampleStruct = struct();
        SampleStruct.R_Real = real(R_equal);
        SampleStruct.R_Imag = imag(R_equal);
        SampleStruct.P_Real = real(P);
        SampleStruct.P_Imag = imag(P);
        SampleStruct.User_Indices = current_users;
        SampleStruct.Noise_Power = noise_power;
        SampleStruct.SNR_dB = snr_db;
        SampleStruct.Group_ID = g_idx;
        if isempty(Group_Samples), Group_Samples = SampleStruct; else, Group_Samples(end+1) = SampleStruct; end
    end
    Results_Cell{g_idx} = Group_Samples;
    
    % 发送进度信号 (关键)
    send(D, g_idx); 
end

% 保存逻辑
disp('正在保存...');
Test_Buffer = [Results_Cell{:}];
save('TestingData_MultiSNR_Slot81to100_20251216.mat', 'Test_Buffer');
fprintf('\n完成！总耗时: %.2f 分钟\n', toc(start_tic)/60);


% === 进度更新 ===
function updateProgress(~)
    persistent count
    if isempty(count), count = 0; end
    count = count + 1;
    
    % 简单的进度条动画
    Num_Groups = 50; % 需与主函数一致，或者通过参数传递
    pct = count / Num_Groups * 50;
    
    % 使用 \b 删除前一次的输出，实现原地刷新
    if count > 1
        fprintf(repmat('\b', 1, 20)); % 这里的20根据下面的打印长度调整
    end
    fprintf('进度: %3.0f%% (%d/%d)', pct, count, Num_Groups);
end