clear all;
close  all;
load('snr_total_Traditional.mat');
cor_total_base1 = cor_total_base(:,3);
distance_base1 = distance_base(:,1);


load('guding_k4_4and20user_1MUL_SNR_cor_total_MMSE_uma_snr_sadian_1.mat');
%load('guding_k4_10user_1MUL_SNR_cor_total_MMSE_uma_snr_sadian_1.mat');
cor_total_base2 = cor_total_MMSE(:,2);
distance_base2 = distance_MMSE(:,1);

load('cor_total_base_uma_sadian_1_snr_20.mat');
cor_total_base3 = cor_total_base(:,3);
distance_base3 = distance_base(:,1);

load('snr_total_EM.mat');
cor_total_base4 = cor_total_base(:,3);

idx = 6;
% 假设x是4到10的值
x = -10:2:20;
figure;
% 获得当前图形窗口的句柄
hFig = gcf;
% 将该图形窗口的背景色设置为灰色
set(hFig, 'Color', [1 1 1]);

% 绘制第一条曲线，使用'o-'代表圆圈标记并且实线连接
%cor_total_base(7,1)=cor_total_base(7,1)*1.1;
tmp = cor_total_base1;
%  tmp(6) = tmp(6) /1.018;
%  tmp(7) = tmp(7) /1.002;
plot(x, tmp, 'o--', 'DisplayName', 'DFDCA','LineWidth', 1);
% 保持当前图形，以便在同一图形上绘制第二条曲线
hold on;

% 绘制第二条曲线，使用'*--'代表星号标记并且虚线连接
tmp = cor_total_base2;
% tmp(6) = tmp(6)/1.02;
% tmp(7) = tmp(7)*1.002;
plot(x, tmp, '*--', 'DisplayName', 'FDCA','LineWidth', 1);
hold on;
tmp = cor_total_base3;
plot(x, tmp, 'd--', 'DisplayName', 'PSOP','LineWidth', 1);

hold on;
tmp = cor_total_base4;
plot(x, tmp, 'x--', 'DisplayName', 'EM','LineWidth', 1);

% 添加图例
legend('FontSize', 12);

% 添加坐标轴标签
xlabel('SNR');
ylabel('MSE');% between the estimated channel and the actual channel

grid on;
% 添加标题
%title('Two Curves on One Plot');

% 取消保持状态，后续图形将在新窗口中绘制
hold off;







