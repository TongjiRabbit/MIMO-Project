clc;
clear all;
close all;
idx_rxtx_diedai_max =1;%%按照文章，迭代3-5次即可
shoulian = zeros(21,4,2,1);
idx_sadian = 0;
for sadian = 1
    idx_sadian = idx_sadian+1;
% 导入通信工具箱
import commsim.*
%load('../user80_uma_6_5.mat'); %<-- 请确保路径正确
load('wcnc_18_40_129_102slot_nloss.mat'); %<-- 请确保路径正确
H_intial2_tmp = H_wcnc(:,:,1:128,1:100);
[Antnum,Usrnum_total,Frenum,Slotnum]=size(H_intial2_tmp);
[D,ID] = DFTMatrix(Frenum);
D_FFT = D;
N = Frenum;

Usrnum_conf =20;%20个一组撒点

%%%移位一下 对到0的位置
WN = exp(-1i*2*pi/Frenum);
k_initial = 0:Frenum-1; 
for usr = 1:Usrnum_conf
    for ant = 1:Antnum
        H_intial2_tmp_power1 = squeeze(H_intial2_tmp(ant,usr+Usrnum_conf*(sadian-1),:,:));
        H_intial2_tmp_power = sqrt(trace(H_intial2_tmp_power1'*H_intial2_tmp_power1)/Frenum/Slotnum);
        if H_intial2_tmp_power > 0
            H_intial2_tmp(ant,usr,:,:) = squeeze(H_intial2_tmp(ant,usr+Usrnum_conf*(sadian-1),:,:))./H_intial2_tmp_power;
        else
             H_intial2_tmp(ant,usr,:,:) = squeeze(H_intial2_tmp(ant,usr+Usrnum_conf*(sadian-1),:,:));
        end
    end
end

idx_snr = 0;
for snr_db =-10:2:20%-10:2:20
idx_snr = idx_snr + 1;
        idx_user = 0;
        for Usrnum = [4,10,20]
            idx_user = idx_user+1;
            
            %%% 1. 分布式系统设置
            K_groups = 4;
            if mod(Antnum, K_groups) ~= 0, error('总天线数必须能被天线组数整除'); end
            Antnum_per_group = Antnum / K_groups;

            k0 = floor(Frenum/Usrnum);
            if Usrnum == 4, k0 = 40; end; if Usrnum == 10, k0 = 16; end; if Usrnum == 15, k0 = 10; end;if Usrnum == 20, k0 = 8; end

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
P = ones(Usrnum,Frenum);
W = ones(Usrnum,k,Frenum, K_groups)/sqrt(Frenum);

%%计算噪声
noise_power_tmp = 0;
for tidx = 5:10:95
    H_tidx = squeeze(H_intial2(:,1:Usrnum,:,tidx));
    signal_power = sum(abs(H_tidx(:)).^2)/numel(H_tidx);
    sol_snr = 10^(snr_db/10);
    noise_power_tmp = noise_power_tmp+sqrt(signal_power/sol_snr);
end
noise_power = noise_power_tmp/10*0.9;


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

for tidx = 5:10:95%第几个slot
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
cor_total_base_tidx(idx_snr,idx_user,tidx)=cor_total_tianxian/Antnum;
distance_base_tidx(idx_snr,idx_user,tidx)=distance_tianxian/Antnum;
end % 时隙结束

        end
    end
end
% 定义文件名的基础部分
base_filename = 'user';
filename = base_filename;
save([filename '.mat'],'cor_total_base_tidx','distance_base_tidx');
cor_total_base=sum(cor_total_base_tidx,3)/10;
distance_base=sum(distance_base_tidx,3)/10;

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