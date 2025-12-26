clear all;
close  all;

% === 加载两组数据（目前都只测试10用户，所以取第1列）===
S1 = load('snr_total_EM_diedaiDFDCA_Evaluation_Results_TrainBatch1to5_test1_4096hidden_500epoch.mat');
S2 = load('snr_total_EM.mat');

user_col = 1; % 只测试10用户时通常只有1列
cor_curve_1 = S1.cor_total_base(:, user_col);
dist_curve_1 = S1.distance_base(:, user_col);
cor_curve_2 = S2.cor_total_base(:, 2);
dist_curve_2 = S2.distance_base(:, 2);

% SNR点与数据生成/评估一致
x = [-10, 0, 5, 10, 15, 20];

% ============ 绘制相关性曲线 ============
figure;
hFig = gcf;
set(hFig, 'Color', [1 1 1]);

plot(x, abs(cor_curve_1), 'o-', 'DisplayName', 'diedai', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
plot(x, abs(cor_curve_2), 's--', 'DisplayName', 'DNN', 'LineWidth', 1.5, 'MarkerSize', 8);
hold off;

legend('FontSize', 12, 'Location', 'best');
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Correlation', 'FontSize', 12);
title('信道恢复相关性 vs SNR', 'FontSize', 14);
grid on;

% ============ 绘制距离曲线 ============
figure;
hFig = gcf;
set(hFig, 'Color', [1 1 1]);

plot(x, abs(dist_curve_1), 'o-', 'DisplayName', 'diedai', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
plot(x, abs(dist_curve_2), 's--', 'DisplayName', 'DNN', 'LineWidth', 1.5, 'MarkerSize', 8);
hold off;

legend('FontSize', 12, 'Location', 'best');
xlabel('SNR (dB)', 'FontSize', 12);
ylabel('Distance', 'FontSize', 12);
title('信道恢复距离 vs SNR', 'FontSize', 14);
grid on;





