clear all;
close  all;

% === 加载 Copy_of_distribute_MMSE_snr_cauculateW_and_reH.m 导出的数据 ===
load('snr_total_EM.mat');  % 包含 cor_total_base, distance_base
% cor_total_base: [6 SNR点 x 3 用户数配置]
% 列1: Usrnum=4, 列2: Usrnum=10, 列3: Usrnum=20

% 选择要绘制的用户数配置（1=4用户, 2=10用户, 3=20用户）
user_col = 2;  % 默认绘制10用户的结果
cor_curve = cor_total_base(:, user_col);
dist_curve = distance_base(:, user_col);

% SNR点与数据生成/评估一致
x = [-10, 0, 5, 10, 15, 20];

% ============ 绘制相关性曲线 ============
figure;
hFig = gcf;
set(hFig, 'Color', [1 1 1]);

plot(x, abs(cor_curve), 'o-', 'DisplayName', 'DNN-P (预测P)', 'LineWidth', 1.5, 'MarkerSize', 8);

legend('FontSize', 12, 'Location', 'best');
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Correlation', 'FontSize', 12);
title('信道恢复相关性 vs SNR', 'FontSize', 14);
grid on;

% ============ 绘制距离曲线 ============
figure;
hFig = gcf;
set(hFig, 'Color', [1 1 1]);

plot(x, abs(dist_curve), 's-', 'DisplayName', 'DNN-P (预测P)', 'LineWidth', 1.5, 'MarkerSize', 8);

legend('FontSize', 12, 'Location', 'best');
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Distance', 'FontSize', 12);
title('信道恢复距离 vs SNR', 'FontSize', 14);
grid on;





