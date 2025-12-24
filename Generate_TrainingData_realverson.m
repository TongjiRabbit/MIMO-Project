%% Generate_TrainingData_2.m
% 功能：生成 DFDCA DNN 训练数据集 (批量保存版)
% 逻辑：采用 C(20,10) 随机组合采样，生成不同的用户干扰模式
% 配置：按每100组用户合并为一个 .mat 文件保存

clc; clear all; close all;

%% 1. 全局配置与数据加载
disp('------------------------------------------------');
disp('正在初始化数据生成环境...');
disp('------------------------------------------------');

% 加载信道数据
dataFileName = 'wcnc_18_40_129_102slot_nloss.mat';
if exist(dataFileName, 'file')
    load(dataFileName); 
else
    error(['错误：找不到文件 ', dataFileName]);
end

% 截取前80个Slot用于训练
Training_Slot_Range = 1:80;
H_source_total = H_wcnc(:, :, :, Training_Slot_Range); 
[Antnum, Total_Pool_Size, Frenum, Slotnum] = size(H_source_total);

% 核心参数
Target_Usrnum = 10;             % 并发用户数
K_groups = 4;                   % 天线分簇数
iter_max = 10;                  % MMSE 迭代次数
SNR_List = [-10, 0, 5, 10, 15, 20]; 

% 建议将此处改为 1000 或 2000 以生成足够数据
Num_Groups = 2000;              % <--- 修改：增加组数
Batch_Size = 100;               % <--- 新增：每100组存一次

User_Groups = zeros(Num_Groups, Target_Usrnum);

fprintf('正在生成 %d 组随机用户组合 (从 %d 个用户中选 %d 个)...\n', ...
    Num_Groups, Total_Pool_Size, Target_Usrnum);

rng('shuffle'); 

for g = 1:Num_Groups
    User_Groups(g, :) = sort(randperm(Total_Pool_Size, Target_Usrnum));
    % 只有第一组打印一下，避免刷屏
    if g==1, fprintf('  Group 1 Users: [%s] (后续省略)\n', num2str(User_Groups(g, :))); end
end

% 基础矩阵准备
[D, ID] = DFTMatrix(Frenum);
WN = exp(-1i*2*pi/Frenum); 
k_initial = 0:Frenum-1; 

if Target_Usrnum == 4, k0 = 40; end
if Target_Usrnum == 10, k0 = 16; end
if Target_Usrnum == 15, k0 = 10; end
if Target_Usrnum == 20, k0 = 8; end

%% 2. 任务队列与监视器初始化
Total_Tasks = Num_Groups * length(SNR_List); 
processed_count = 0;
start_tic = tic;
Batch_Buffer = []; % <--- 新增：初始化缓冲区

fprintf('\n开始并行优化计算... 总任务数: %d\n', Total_Tasks);
fprintf('|%-10s|%-10s|%-10s|%-15s|%-15s|\n', 'Group', 'SNR(dB)', 'Progress', 'Elapsed', 'Remaining');
fprintf('------------------------------------------------------------------\n');

%% 3. 主循环处理
for g_idx = 1:Num_Groups
    current_users = User_Groups(g_idx, :);
    
    % === A. 信道提取与归一化 ===
    H_intial2_tmp = zeros(Antnum, Target_Usrnum, Frenum, Slotnum);
    for u_local = 1:Target_Usrnum
        u_global = current_users(u_local);
        H_intial2_tmp(:, u_local, :, :) = H_source_total(:, u_global, :, :);
    end
    
    for usr = 1:Target_Usrnum
        for ant = 1:Antnum
            H_vec = squeeze(H_intial2_tmp(ant, usr, :, :));
            H_pwr = sqrt(trace(H_vec' * H_vec) / Frenum / Slotnum);
            if H_pwr > 0
                H_intial2_tmp(ant, usr, :, :) = H_vec ./ H_pwr;
            end
        end
    end
    
    H_intial2 = zeros(Antnum, Target_Usrnum, Frenum, Slotnum);
    for ant = 1:Antnum
        for usr = 1:Target_Usrnum
            D2 = WN .^ (k_initial * (usr-1) * k0);
            H_intial2(ant, usr, :, :) = transpose(D2) .* squeeze(H_intial2_tmp(ant, usr, :, :));
        end
    end
    
    % === B. 统计特征提取 ===
    Antnum_per_group = Antnum / K_groups;
    R = zeros(Frenum, Frenum, Target_Usrnum, K_groups);
    R_central_pro = zeros(Frenum, Frenum, Target_Usrnum, K_groups);
    R_equal = zeros(Frenum, 4, Target_Usrnum, K_groups); % 预分配
    
    for k_group = 1:K_groups
        ant_start = (k_group - 1) * Antnum_per_group + 1; 
        ant_end = k_group * Antnum_per_group;
        for usridx = 1:Target_Usrnum
            for antidx = ant_start:ant_end
                for slotidx = 1:Slotnum
                    H_tmp = squeeze(H_intial2(antidx, usridx, :, slotidx));
                    R(:,:,usridx,k_group) = R(:,:,usridx,k_group) + H_tmp * H_tmp';
                end
            end
        end
    end
    R = R ./ (Antnum_per_group * Slotnum);
    
    k = 4; k_central_pro = 6;
    U = zeros(Frenum, k, Target_Usrnum, K_groups);
    
    for usridx = 1:Target_Usrnum
        for k_group = 1:K_groups
            [U_tmp, S_tmp, ~] = svd(R(:,:,usridx,k_group)); 
            U(:,:,usridx,k_group) = U_tmp(:, 1:k);
            % 这里保留了 U*sqrt(S) 的逻辑
            R_equal(:,:,usridx,k_group) = U_tmp(:, 1:k)*sqrt(S_tmp(1:k,1:k));
            R_central_pro(:,:,usridx,k_group) = U_tmp(:,1:k_central_pro)*S_tmp(1:k_central_pro,1:k_central_pro)*U_tmp(:,1:k_central_pro)';
        end
    end
    
    c_2 = zeros(k, Target_Usrnum, K_groups);
    for usridx = 1:Target_Usrnum
        for k_group = 1:K_groups
            tmpU1 = squeeze(U(:,:,usridx,k_group)); 
            c_tmp = zeros(k, 1);
            ant_start = (k_group-1)*Antnum_per_group+1; 
            ant_end = k_group*Antnum_per_group;
            for antidx = ant_start:ant_end
                for slotidx = 1:Slotnum
                    H_tmp = squeeze(H_intial2(antidx, usridx, :, slotidx)); 
                    c_tmp_tmp = tmpU1' * H_tmp;
                    for k_idx = 1:k, c_tmp(k_idx) = c_tmp(k_idx) + c_tmp_tmp(k_idx) * c_tmp_tmp(k_idx)'; end
                end
            end
            for k_idx = 1:k, c_2(k_idx,usridx,k_group) = c_tmp(k_idx)/(Antnum_per_group*Slotnum); end
        end
    end
    
    % === C. SNR 循环 ===
    for s_idx = 1:length(SNR_List)
        snr_db = SNR_List(s_idx);
        processed_count = processed_count + 1;
        
        noise_power_tmp = 0;
        sample_slots = 5:10:75; 
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
            % ... Receiver Update (代码保持不变) ...
            for k_group = 1:K_groups
                SSH_total_k = zeros(Frenum, Frenum);
                for i = 1:Target_Usrnum
                    tmpP1 = squeeze(P(i,:)); tmpP2 = diag(tmpP1); 
                    tmpR = squeeze(R(:,:,i,k_group)); 
                    SSH_total_k = SSH_total_k + tmpP2 * tmpR * tmpP2'; 
                end
                Sc_k = zeros(Frenum, k, Target_Usrnum);
                for i = 1:Target_Usrnum
                    for j = 1:k
                        X_k = squeeze(U(:,j,i,k_group)) .* transpose(P(i,:)); 
                        Sc_k(:,j,i) = X_k .* c_2(j,i,k_group); 
                    end
                end
                matrix_inv = pinv(SSH_total_k + noise_power^2*eye(Frenum) + 1e-9*eye(Frenum));
                for i = 1:Target_Usrnum
                    for j = 1:k, W(i,j,:,k_group) = matrix_inv * Sc_k(:,j,i); end
                end
            end
            
            % ... Transmitter Update (代码保持不变) ...
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
        
        % === D. [修改] 存入缓冲区而不是直接保存 ===
        SampleStruct.R_Real = real(R_equal);
        SampleStruct.R_Imag = imag(R_equal);
        SampleStruct.P_Real = real(P);
        SampleStruct.P_Imag = imag(P);
        SampleStruct.User_Indices = current_users;
        SampleStruct.Noise_Power = noise_power;
        SampleStruct.SNR_dB = snr_db;     % 新增：记录当前SNR
        SampleStruct.Group_ID = g_idx;    % 新增：记录当前Group ID
        
        % 将单条数据追加到 Buffer
        if isempty(Batch_Buffer)
            Batch_Buffer = SampleStruct;
        else
            Batch_Buffer(end+1) = SampleStruct;
        end
        
        % === E. 监视器更新 ===
        time_elapsed = toc(start_tic);
        avg_time = time_elapsed / processed_count;
        remain_tasks = Total_Tasks - processed_count;
        time_remain = avg_time * remain_tasks;
        str_elap = sprintf('%02dm %02ds', floor(time_elapsed/60), floor(mod(time_elapsed, 60)));
        str_rem  = sprintf('%02dm %02ds', floor(time_remain/60), floor(mod(time_remain, 60)));
        progress_pct = (processed_count / Total_Tasks) * 100;
        
        % 打印状态 (每10个进度打印一次，避免太快)
        if mod(processed_count, 10) == 0 || processed_count == Total_Tasks
             fprintf('|   %-6d |   %-6d  |   %5.1f%%   | %-13s | %-13s |\n', ...
            g_idx, snr_db, progress_pct, str_elap, str_rem);
        end
        
    end % SNR Loop

    % === [新增] 批量保存逻辑 ===
    % 如果达到100组，或者已经是最后一组，就保存并清空缓冲区
    if mod(g_idx, Batch_Size) == 0 || g_idx == Num_Groups
        batch_id = ceil(g_idx / Batch_Size);
        save_filename = sprintf('TrainData_Batch_%d.mat', batch_id);
        
        fprintf('   >>> [保存] 正在写入第 %d 批数据 (%s)...\n', batch_id, save_filename);
        save(save_filename, 'Batch_Buffer');
        
        % 清空缓冲区，释放内存
        Batch_Buffer = []; 
    end
    
end % Group Loop

fprintf('------------------------------------------------------------------\n');
fprintf('所有任务完成！总耗时: %.2f 分钟\n', toc(start_tic)/60);