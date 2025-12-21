function [lamda_p] = lamdacal(tmp_ip,tmp_sc,Frenum)

% 开始和结束界限
a = 0.000001;
b = 2000;
tol = 0.00000001; % 定义需要的解的精确度

% 开始二分法
while (b - a) / 2 > tol
    c = (a + b) / 2;
    if ff(tmp_ip,tmp_sc,Frenum,c)-Frenum == 0
        break; % 解已经找到，c 就是解
    elseif (ff(tmp_ip,tmp_sc,Frenum,a)-Frenum)*(ff(tmp_ip,tmp_sc,Frenum,c)-Frenum) < 0
        b = c; % 新界限是 [a, c]
    else
        a = c; % 新界限是 [c, b]
    end
    if ((ff(tmp_ip,tmp_sc,Frenum,a)-Frenum)<0) && ((ff(tmp_ip,tmp_sc,Frenum,b)-Frenum)<0)
        break;
    end
end

% 现在a和b够接近了，我们可以认为解大约就是它们的平均值
lamda_p = (a + b) / 2;

end