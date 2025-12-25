clc;
clear all;
close all;
idx_rxtx_diedai_max =1;%%按照文章，迭代3-5次即可
shoulian = zeros(21,4,2,1);
idx_sadian = 0;

% === 新增：从 Evaluate 导出的 .mat 读取预测 P（不再迭代更新P）===
EVAL_P_MAT_FILE = fullfile('Result_P', 'DFDCA_Evaluation_Results_TrainBatch1to5_test1_4096hidden_500epoch.mat');
if ~exist(EVAL_P_MAT_FILE, 'file')
    error(['找不到 Evaluate 导出的结果文件: ', EVAL_P_MAT_FILE]);
end
tmp_eval = load(EVAL_P_MAT_FILE, 'Batch_Buffer');
Eval_Buffer = tmp_eval.Batch_Buffer;
clear tmp_eval;

for sadian = 1
    idx_sadian = idx_sadian+1;
% 导入通信工具箱
import commsim.*
CHANNEL_FILE = 'D:\CodeSpace\CodeOfMMSE_xUser_re_OnlyTrainToTrain\Data\wcnc_18_40_129_102slot_nloss.mat';%注意路径配置
load(CHANNEL_FILE);
[Antnum,Usrnum_total,~,~]=size(H_wcnc);
Frenum = 128;
Slotnum = 80;%测试集只选了前80个slot
[D,ID] = DFTMatrix(Frenum);
D_FFT = D;
N = Frenum;

Usrnum_conf =20;%20个一组撒点（原逻辑保留，但本脚本将按 Batch_Buffer 的 User_Indices 动态选取用户）

idx_snr = 0;
% SNR 仅评估数据中实际存在的点（保持与数据生成/评估一致）
for snr_db = [-10, 0, 5, 10, 15, 20]
idx_snr = idx_snr + 1;
        idx_user = 0;
        for Usrnum = [4,10,20]
            idx_user = idx_user+1;

            % 仅把“迭代得到的P”替换为“从 .mat 的 Batch_Buffer 取预测P”
            % 本 Evaluate 数据通常是固定并发用户数（如10），其余用户数直接跳过
            eval_mask = false(1, numel(Eval_Buffer));
            for ii = 1:numel(Eval_Buffer)
                snr_tmp = Eval_Buffer(ii).SNR_dB;
                if numel(snr_tmp) > 1, snr_tmp = snr_tmp(1); end
                usr_idx_tmp = Eval_Buffer(ii).User_Indices;
                eval_mask(ii) = (round(double(snr_tmp)) == snr_db) && (numel(usr_idx_tmp) == Usrnum);
            end
            eval_indices = find(eval_mask);
            if isempty(eval_indices)
                continue;
            end
            
            %%% 1. 分布式系统设置
            K_groups = 4;
            if mod(Antnum, K_groups) ~= 0, error('总天线数必须能被天线组数整除'); end
            Antnum_per_group = Antnum / K_groups;

            k0 = floor(Frenum/Usrnum);
            if Usrnum == 4, k0 = 40; end; if Usrnum == 10, k0 = 16; end; if Usrnum == 15, k0 = 10; end;if Usrnum == 20, k0 = 8; end

            % 评估的 slot 列表（与下方性能评估循环保持一致）
            tidx_list = 5:10:75;

            % === 修复：对同一 (SNR,Usrnum) 下的多个样本做平均，避免 idx_eval 覆盖结果 ===
            for tt = tidx_list
                cor_total_base_tidx(idx_snr,idx_user,tt) = 0;
                distance_base_tidx(idx_snr,idx_user,tt) = 0;
            end
            eval_count = 0;

            % 对每个样本（每组用户）分别评估：按 User_Indices 取H、用预测P算W、再恢复信道
            for idx_eval = 1:numel(eval_indices)
                bb = Eval_Buffer(eval_indices(idx_eval));
                current_users = double(bb.User_Indices(:).');

                % === A. 根据索引从 H_wcnc 取对应用户，并做归一化（沿用原归一化方式）===
                H_intial2_tmp = zeros(Antnum,Usrnum,Frenum,Slotnum);
                for u_local = 1:Usrnum
                    u_global = current_users(u_local);
                    H_intial2_tmp(:,u_local,:,:) = H_wcnc(:,u_global,1:Frenum,1:Slotnum);
                end
                for usr = 1:Usrnum
                    for ant = 1:Antnum
                        H_intial2_tmp_power1 = squeeze(H_intial2_tmp(ant,usr,:,:));
                        H_intial2_tmp_power = sqrt(trace(H_intial2_tmp_power1'*H_intial2_tmp_power1)/Frenum/Slotnum);
                        if H_intial2_tmp_power > 0
                            H_intial2_tmp(ant,usr,:,:) = squeeze(H_intial2_tmp(ant,usr,:,:))./H_intial2_tmp_power;
                        end
                    end
                end

                % === B. 移位 (对到0的位置)（原逻辑保留）===
                k_initial = 0:Frenum-1; 
                WN = exp(-1i*2*pi/Frenum);
                H_intial2 = zeros(Antnum, Usrnum, Frenum, Slotnum);
                for ant = 1:Antnum
                    for usr = 1:Usrnum
                        D2 = WN.^ (k_initial*(usr-1)*k0);
                        H_intial2(ant,usr,:,:) = transpose(D2).*squeeze(H_intial2_tmp(ant,usr,:,:));
                    end
                end

%%求统计信道                   
%%% 2. R, U, c_2 本地化 (此段逻辑已验证正确)
R = zeros(Frenum,Frenum,Usrnum, K_groups);
R_central_pro = zeros(Frenum,Frenum,Usrnum, K_groups);
k_central_pro = 6;
for k_group = 1:K_groups
    ant_start = (k_group - 1) * Antnum_per_group + 1; ant_end = k_group * Antnum_per_group;
    for usridx = 1:Usrnum, for antidx = ant_start:ant_end, for slotidx = 1:Slotnum
        H_tmp = squeeze(H_intial2(antidx,usridx,:,slotidx));
        R(:,:,usridx,k_group) = R(:,:,usridx,k_group)+H_tmp*H_tmp';
    end, end, end
end
R = R./(Antnum_per_group*Slotnum);

N = Frenum;
    if(Usrnum<=4), k = 7; elseif(Usrnum>4 &&Usrnum<=8), k = 6; elseif(Usrnum>8 &&Usrnum<=12), k = 5; else, k = 4; end
    k=4;
    U = zeros(Frenum,k,Usrnum, K_groups);
for usridx = 1:Usrnum
    for k_group = 1:K_groups
        [U_tmp,S_tmp,~]=svd(R(:,:,usridx,k_group)); 
        U(:,:,usridx,k_group) = U_tmp(:,1:k);
        R_central_pro(:,:,usridx,k_group) = U_tmp(:,1:k_central_pro)*S_tmp(1:k_central_pro,1:k_central_pro)*U_tmp(:,1:k_central_pro)';
    end
end
c_2 = zeros(k,Usrnum, K_groups);
for usridx = 1:Usrnum, for k_group = 1:K_groups
    tmpU1 = squeeze(U(:,:,usridx,k_group)); c_tmp = zeros(k,1);
    ant_start = (k_group-1)*Antnum_per_group+1; ant_end = k_group*Antnum_per_group;
    for antidx=ant_start:ant_end, for slotidx=1:Slotnum, H_tmp=squeeze(H_intial2(antidx,usridx,:,slotidx)); c_tmp_tmp=tmpU1'*H_tmp;
        for k_idx=1:k, c_tmp(k_idx)=c_tmp(k_idx)+c_tmp_tmp(k_idx)*c_tmp_tmp(k_idx)'; end
    end, end
    for k_idx=1:k, c_2(k_idx,usridx,k_group)=c_tmp(k_idx)/(Antnum_per_group*Slotnum); end
end, end

%%初始化
% P：直接使用 Evaluate 导出的预测P（不再迭代更新P）
if isfield(bb, 'P_Real_Pred') && isfield(bb, 'P_Imag_Pred')
    P = bb.P_Real_Pred + 1i*bb.P_Imag_Pred;
else
    error('Batch_Buffer 中缺少 P_Real_Pred/P_Imag_Pred 字段，请确认 Evaluate 导出格式。');
end
W = ones(Usrnum,k,Frenum, K_groups)/sqrt(Frenum);

%%计算噪声
noise_power_tmp = 0;
for tidx = 5:10:75
    H_tidx = squeeze(H_intial2(:,1:Usrnum,:,tidx));
    signal_power = sum(abs(H_tidx(:)).^2)/numel(H_tidx);
    sol_snr = 10^(snr_db/10);
    noise_power_tmp = noise_power_tmp+sqrt(signal_power/sol_snr);
end
noise_power = noise_power_tmp/length(5:10:75)*0.9;


for idx_rxtx_diedai = 1:idx_rxtx_diedai_max

    % 保存shoulian的逻辑现在统一使用辅助函数，避免代码重复
    calculate_and_save_mse(idx_rxtx_diedai*2-1, P,W,R,U,c_2,k,noise_power,K_groups,Frenum,Usrnum,shoulian,idx_user,idx_snr,idx_sadian);

%接收端 (此段逻辑已验证正确)
for k_group = 1:K_groups
    SSH_total_k = zeros(Frenum,Frenum);
    for i=1:Usrnum, tmpP1=squeeze(P(i,:)); tmpP2=diag(tmpP1); tmpR=squeeze(R(:,:,i,k_group)); SSH_total_k=SSH_total_k+tmpP2*tmpR*tmpP2'; end
    Sc_k = zeros(Frenum,k,Usrnum);
    for i=1:Usrnum, for j=1:k, X_k=squeeze(U(:,j,i,k_group)).*transpose(P(i,:)); Sc_k(:,j,i)=X_k.*c_2(j,i,k_group); end, end
    for i = 1:Usrnum, for j = 1:k
        matrix_to_invert_W = SSH_total_k+noise_power^2*eye(Frenum) + 1e-9*eye(Frenum);
        W(i,j,:,k_group) = pinv(matrix_to_invert_W)*Sc_k(:,j,i);
    end, end
end

    calculate_and_save_mse(idx_rxtx_diedai*2, P,W,R,U,c_2,k,noise_power,K_groups,Frenum,Usrnum,shoulian,idx_user,idx_snr,idx_sadian);

%发送端
%发送端
%发送端
if(idx_rxtx_diedai<idx_rxtx_diedai_max)
    %%% MODIFICATION START: P 更新 (最终决定版，采用清晰独立的计算循环)

    % 对每个用户i，独立地、完整地计算其P更新所需的A_i和b_i
    for i_target = 1:Usrnum
        
        % --- 第1步: 为目标用户'i_target'计算其全局近似R* ---
        %R_approx_conj = sum(conj(squeeze(R(:,:,i_target,:))), 3);

        % --- 第2步: 计算目标用户'i_target'的A矩阵 ---
        % (严格遵循您原始代码的公式: diag(W)*R_conj*diag(W)')
        A_i = zeros(Frenum, Frenum);
        for m_sum = 1:Usrnum      % 求和索引: 遍历所有用户 m
            for j_sum = 1:k       % 求和索引: 遍历所有基 j
                for k_sum = 1:K_groups % 求和索引: 遍历所有天线组 k
                    W_vec = squeeze(W(m_sum, j_sum, :, k_sum));
                    R_approx_conj_tmp = conj(squeeze(R_central_pro(:,:,i_target,k_sum)));
                    %R_approx_conj_tmp = conj(squeeze(R(:,:,i_target,k_sum)));
                    A_i = A_i + diag(W_vec) * R_approx_conj_tmp * diag(W_vec)';
                end
            end
        end

        % --- 第3步: 计算目标用户'i_target'的b向量 ---
        % (严格遵循您原始代码的公式: D'*W*c2)
        b_i = zeros(Frenum, 1);
        for j_sum = 1:k           % 求和索引: 遍历所有基 j
            for k_sum = 1:K_groups % 求和索引: 遍历所有天线组 k
                U_vec = squeeze(U(:, j_sum, i_target, k_sum));
                W_vec = squeeze(W(i_target, j_sum, :, k_sum));
                c2_val = c_2(j_sum, i_target, k_sum);
                b_i = b_i + diag(U_vec)' * (W_vec) * c2_val;
            end
        end

        % --- 第4步: 使用正确的A_i和b_i更新P(i_target,:) ---
        tmp_ip = A_i;
        tmp_sc = b_i;
        
        p_unconstrained = pinv(tmp_ip + 1e-9*eye(Frenum)) * tmp_sc;
        power_unconstrained = p_unconstrained' * p_unconstrained;
        
        if power_unconstrained > Frenum
            lamda_p = lamdacal(tmp_ip,tmp_sc,Frenum);
        else
            lamda_p = 0;
        end
        
        matrix_to_invert_P = tmp_ip+lamda_p*eye(Frenum) + 1e-9*eye(Frenum);
        tmp_p = pinv(matrix_to_invert_P)*tmp_sc;
        power_p = tmp_p'*tmp_p;
        if power_p > 1e-9, P(i_target,:) = tmp_p * sqrt(Frenum/power_p); end
    end
    %%% MODIFICATION END
end
    %calculate_and_save_mse(idx_rxtx_diedai*2+1, P,W,R,U,c_2,k,noise_power,K_groups,Frenum,Usrnum,shoulian,idx_user,idx_snr,idx_sadian);
    %shoulian
end
%%收发迭代over

%%%计算性能

for tidx = 5:10:75%第几个slot
cor_total_tianxian = 0;
distance_tianxian = 0;

H_tidx = squeeze(H_intial2(:,1:Usrnum,:,tidx));
R_tidx2 = H_tidx.*conj(H_tidx);
signal_power = sum(sum(sum(R_tidx2)))/Frenum/Antnum/Usrnum;
sol_snr = 10^(snr_db/10);  %10*log10(x)=snr  x = exp(snr/10)
noise_power = sqrt(signal_power/sol_snr);
for tianxian = 1:Antnum
% 创建导频信号
t = 0:1/N:1-1/N;  % 时间轴
inputSig_f =ones(N,1);%全1的输入信号
inputSig = D_FFT*inputSig_f;

channel_total_tmp = squeeze(H_intial2(tianxian,1:Usrnum,:,tidx));
channel_total = permute(channel_total_tmp,[2,1]);

%%过信道
for usridx = 1:Usrnum
    channel_vector_f_usr(:,usridx) = channel_total(:,usridx);
end

channel_vector = zeros(N,1);
for idx = 1:Usrnum
    tmpP = transpose(squeeze(P(idx,:)));
    channel_vector = channel_vector+channel_total(:,idx).*tmpP;
end
channel_vector = ID*channel_vector;
channel_vector_f = D_FFT*channel_vector;


 % 生成两个独立的标准正态分布随机变量
    realPart = randn(size(inputSig_f));
    imagPart = randn(size(inputSig_f));
    % 转换为复数
    complexNumber = realPart + 1i * imagPart;
    % 标准化模长为 1
    normalizedComplexNumber = complexNumber ./ sqrt(2);
    normalizedComplexNumber = complexNumber ./ abs(complexNumber);
    noise = noise_power*normalizedComplexNumber; 

    channel_vector_f = channel_vector_f+noise;%add 噪声

channel_vector = ID*channel_vector_f;


outputSig_f = channel_vector_f.*inputSig_f;%输出的频域信号


k_group = floor((tianxian - 1) / Antnum_per_group) + 1;
%%计算特征系数  恢复信道
c = zeros(Usrnum,k); % 初始化
for i = 1:Usrnum
    for j = 1:k
            % MODIFIED: 從4維的W矩陣中，選取隸屬當前天線組 k_group 的濾波器
            w_vec = squeeze(W(i,j,:,k_group));
            c(i,j) = w_vec' * outputSig_f;
            A=1;
    end
end
%% 恢復通道 (分布式修改版)
outputSig_f_tezheng = zeros(N,Usrnum);
for i = 1:Usrnum
    for j = 1:k
        % MODIFIED: 從4維的U矩陣中，選取隸屬當前天線組 k_group 的特徵基
        u_vec = squeeze(U(:,j,i,k_group));
        outputSig_f_tezheng(:,i) = outputSig_f_tezheng(:,i) + c(i,j) * u_vec;
    end
end

cor = zeros(Usrnum,1);
for usridx = 1:Usrnum
    cor(usridx) =  cor(usridx)+outputSig_f_tezheng(:,usridx)'*channel_vector_f_usr(:,usridx)/sqrt(outputSig_f_tezheng(:,usridx)'*outputSig_f_tezheng(:,usridx))/sqrt(channel_vector_f_usr(:,usridx)'*channel_vector_f_usr(:,usridx));
end

distance=zeros(Usrnum,1);
for usridx = 1:Usrnum
    A_est =outputSig_f_tezheng(:,usridx)/sqrt(outputSig_f_tezheng(:,usridx)'*outputSig_f_tezheng(:,usridx));
    A_true = channel_vector_f_usr(:,usridx)/sqrt(channel_vector_f_usr(:,usridx)'*channel_vector_f_usr(:,usridx));
   distance(usridx)= sum(abs(A_est - A_true).^2);
   distance(usridx)= distance(usridx)/1;
end
cor_total_tianxian=cor_total_tianxian+sum(abs(cor))/Usrnum;
distance_tianxian=distance_tianxian+sum(abs(distance))/Usrnum;
end% tianxian结束
abs(cor);
cor_total_base_tidx(idx_snr,idx_user,tidx)=cor_total_base_tidx(idx_snr,idx_user,tidx)+cor_total_tianxian/Antnum;
distance_base_tidx(idx_snr,idx_user,tidx)=distance_base_tidx(idx_snr,idx_user,tidx)+distance_tianxian/Antnum;
end % 时隙结束

                eval_count = eval_count + 1;

            end % idx_eval

            % 将累加结果取平均
            if eval_count > 0
                for tt = tidx_list
                    cor_total_base_tidx(idx_snr,idx_user,tt) = cor_total_base_tidx(idx_snr,idx_user,tt) / eval_count;
                    distance_base_tidx(idx_snr,idx_user,tt) = distance_base_tidx(idx_snr,idx_user,tt) / eval_count;
                end
            end

        end
    end
end
% 定义文件名的基础部分
base_filename = 'user';
filename = base_filename;
save([filename '.mat'],'cor_total_base_tidx','distance_base_tidx');
cor_total_base=sum(cor_total_base_tidx,3)/numel(5:10:75);
distance_base=sum(distance_base_tidx,3)/numel(5:10:75);

base_filename = 'snr_total_EM';
filename = base_filename;
save([filename '.mat'],'cor_total_base','distance_base');

% 辅助函数，用于计算和保存MSE，避免主循环代码冗长
function calculate_and_save_mse(index, P,W,R,U,c_2,k,noise_power,K_groups,Frenum,Usrnum,shoulian_ref,idx_user,idx_snr,idx_sadian)
    total_mse = 0;
    total_c2 = 0;
    for k_group = 1:K_groups
        SSH_k=zeros(Frenum,Frenum);
        for i_usr=1:Usrnum, p_vec=squeeze(P(i_usr,:)); p_mat=diag(p_vec); r_mat=squeeze(R(:,:,i_usr,k_group)); SSH_k=SSH_k+p_mat*r_mat*p_mat'; end
        
        mse_k = zeros(Usrnum,k);
        for i_usr=1:Usrnum
            for j_k=1:k
                X_vec = squeeze(U(:,j_k,i_usr,k_group)).*transpose(P(i_usr,:));
                Sc_vec = X_vec .* c_2(j_k,i_usr,k_group);
                % 根据index是奇数还是偶数，判断是r2还是r1的计算
%                 if mod(index, 2) == 1 % 对应r2的计算 (基于X)
%                     mse_k(i_usr,j_k) = X_vec'*(SSH_k+noise_power^2*eye(Frenum))*X_vec - X_vec'*Sc_vec - Sc_vec'*X_vec + c_2(j_k,i_usr,k_group);
%                 else % 对应r1的计算 (基于W)
%                     w_vec = squeeze(W(i_usr,j_k,:,k_group));
%                     mse_k(i_usr,j_k) = w_vec'*(SSH_k+noise_power^2*eye(Frenum))*w_vec-w_vec'*Sc_vec-Sc_vec'*w_vec+c_2(j_k,i_usr,k_group);
%                 end
                    w_vec = squeeze(W(i_usr,j_k,:,k_group));
                    mse_k(i_usr,j_k) = w_vec'*(SSH_k+noise_power^2*eye(Frenum))*w_vec-w_vec'*Sc_vec-Sc_vec'*w_vec+c_2(j_k,i_usr,k_group);
            end
        end
        total_mse = total_mse + sum(sum(mse_k));
        total_c2 = total_c2 + sum(sum(c_2(:,:,k_group)));
    end
    shoulian_ref(index,idx_user,idx_snr,idx_sadian) = total_mse / total_c2;
    % 将结果写回主工作区
    assignin('base', 'shoulian', shoulian_ref);
end