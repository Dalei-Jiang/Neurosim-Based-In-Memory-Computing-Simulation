clc;
clear;
close all;

exp_LTP_raw = load('./weight/LTP.mat').exp_LTP_raw;
exp_LTD_raw = load('./weight/LTD.mat').exp_LTD_raw;

xf_ltp = floor(max(exp_LTP_raw(:,1)));
xf_ltd = floor(max(exp_LTD_raw(:,1)));

x_step_ltp = 1/xf_ltp;
x_step_ltd = 1/xf_ltd;

yf_ltp = max(exp_LTP_raw(:,2));
yf_ltd = max(exp_LTD_raw(:,2));
yi_ltp = min(exp_LTP_raw(:,2));
yi_ltd = min(exp_LTD_raw(:,2));

exp_LTP(:,1) = exp_LTP_raw(:,1)/xf_ltp;
exp_LTD(:,1) = exp_LTD_raw(:,1)/xf_ltd;
exp_LTP(:,2) = (exp_LTP_raw(:,2)-yi_ltp)/(yf_ltp-yi_ltp);
exp_LTD(:,2) = (exp_LTD_raw(:,2)-yi_ltd)/(yf_ltd-yi_ltd);

plot(exp_LTP(:,1), exp_LTP(:,2), 'bo', 'LineWidth', 1);
hold on;
plot(exp_LTD(:,1), exp_LTD(:,2), 'ro', 'LineWidth', 1);
xf = 1;

A_LTP = 0.5;
B_LTP = 1./(1-exp(-1./A_LTP));
A_LTD = -0.2;
B_LTD = 1./(1-exp(-1./A_LTD));

% LTP fitting
var_amp = 0.035;    % LTP cycle-to-cycle variation
rng(103);
x_ltp(1) = 0;
y_ltp(1) = 0;

for n=1:1/x_step_ltp+1
    x_ltp(n+1) = x_ltp(n)+x_step_ltp;
    y_ltp(n+1) = B_LTP(1)*(1-exp(-x_ltp(n+1)/A_LTP(1)));
    delta_y = (y_ltp(n+1)-y_ltp(n)) + randn*var_amp;
    y_ltp(n+1) = y_ltp(n) + delta_y;   
    if y_ltp(n+1)>=1
        y_ltp(n+1)=1;
    elseif y_ltp(n+1)<=0
        y_ltp(n+1)=0;
    end
    x_ltp(n+1) = -A_LTP(1)*log(1-(y_ltp(n+1))/B_LTP(1));
end
plot((0:n-1)/(n-1), y_ltp(1:n), 'b', 'linewidth', 2);

% LTD fitting
var_amp = 0.025;    % LTD cycle-to-cycle variation
rng(898);
x_ltd(1) = 1;
y_ltd(1) = 1;
for n=1:1/x_step_ltd+1
    x_ltd(n+1) = x_ltd(n)-x_step_ltd;
    y_ltd(n+1) = B_LTD(1)*(1-exp(-x_ltd(n+1)/A_LTD(1)));
    delta_y = (y_ltd(n+1)-y_ltd(n)) + randn*var_amp;
    y_ltd(n+1) = y_ltd(n) + delta_y;
    if y_ltd(n+1)>=1
        y_ltd(n+1)=1;
    elseif y_ltd(n+1)<=0
        y_ltd(n+1)=0;
    end
    x_ltd(n+1) = -A_LTD(1)*log(1-(y_ltd(n+1))/B_LTD(1));
end

x_start = numel(x_ltd(:));
x_end = numel(x_ltd(:)) - n;
plot((n-1:-1:0)/(n-1), y_ltd(1:n), 'r', 'linewidth', 2);

xlabel('Normalized Pulse #');
ylabel('Normalized Conductance');
legend('Exp data (LTP)','Exp data (LTD)', 'Fit (LTP)', 'Fit (LTD)', 'location', 'southeast');