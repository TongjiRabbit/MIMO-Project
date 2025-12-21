function [D,ID] = DFTMatrix(Frenum)
%UNTITLED 此处提供此函数的摘要
%   此处提供详细说明
% 生成N点DFT矩阵
N = Frenum; % 点数
k = 0:N-1; n = 0:N-1;
WN = exp(-1i*2*pi/N);
nk = n'*k;
D = WN .^ nk / sqrt(N);
ID = D';
end