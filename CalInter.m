function [result] = CalInter(Omiga_pi,Omiga_pi_inter)
%UNTITLED5 此处提供此函数的摘要
%   此处提供详细说明
fenzi = sum(sum(Omiga_pi.*Omiga_pi_inter));
fenmu1 = sum(sum(Omiga_pi.*Omiga_pi));
fenmu2 = sum(sum(Omiga_pi_inter.*Omiga_pi_inter));
result = fenzi/sqrt(fenmu1)/sqrt(fenmu2);
end