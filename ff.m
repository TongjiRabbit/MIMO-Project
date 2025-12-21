function [shuzhi] = ff(tmp_ip,tmp_sc,Frenum,lamda_p)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
shuzhi=(pinv(tmp_ip+lamda_p*diag(ones(Frenum,1)))*tmp_sc)'*pinv(tmp_ip+lamda_p*diag(ones(Frenum,1)))*tmp_sc;
end