clc;
clear;
close all;
LTP = load('./weight/LTP.mat').LTP;
starting_LTP = min(LTP);
LTP_normal = LTP;
LTP_normal(:,1) = LTP(:,1) - starting_LTP(1,1);
LTP_normal(:,2) = LTP(:,2) - starting_LTP(1,2);
ending_LTP = max(LTP_normal);
max_level_LTP = ending_LTP(1,1);
max_current_LTP = ending_LTP(1,2);
LTP_normal(:,1) = LTP_normal(:,1)/max_level_LTP;
LTP_normal(:,2) = LTP_normal(:,2)/max_current_LTP;
FT=fittype('(1-exp(-x/A))/(1-exp(-1/A))');
opts = fitoptions('Method', 'NonlinearLeastSquares', ...
                  'StartPoint', 1, ...
                  'Lower', 0.01, ...   
                  'Upper', 100, ...     
                  'MaxFunEvals', 1000, ... 
                  'MaxIter', 1000, ...   
                  'TolFun', 1e-6, ...   
                  'TolX', 1e-6);
LTP_FT=fit(LTP_normal(:,1),LTP_normal(:,2),FT,opts);
LTP_A = LTP_FT.A;
LTP_fit = (1 - exp(-LTP_normal(:,1) / LTP_A)) / (1 - exp(-1 / LTP_A));

LTD = load('./weight/LTD.mat').LTD;
LTD_normal = LTD;
max_level = max(LTD);
max_current = max_level(1,2);
LTD_normal(:,2) = max_current - LTD_normal(:,2);
min_level = min(LTD);
LTD_normal(:,1) = LTD_normal(:,1) - min_level(1,1);
max_level = max(LTD_normal);
LTD_normal(:,1) = LTD_normal(:,1) / max_level(1,1);
LTD_normal(:,2) = LTD_normal(:,2) / max_level(1,2);
LTD_FT=fit(LTD_normal(:,1),LTD_normal(:,2),FT,opts);
LTD_A = LTD_FT.A;
LTD_fit = (1 - exp(-LTD_normal(:,1) / LTD_A)) / (1 - exp(-1 / LTD_A));

LTP_normal_x = LTP_normal(:,1);
LTP_normal_y_exp = LTP_normal(:,2);
LTP_normal_y_fit = LTP_fit;
LTD_normal_x = LTD_normal(:,1);
LTD_normal_y_exp = 1.0-LTD_normal(:,2);
LTD_normal_y_fit = 1.0-LTD_fit;

LTP_normal_y_fit_unnormalize = LTP_normal_y_fit*ending_LTP(1,2)+starting_LTP(1,2);
LTD_normal_y_fit_unnormalize = (max_current - LTD_fit*max_level(1,2));


% N = 64;
% hold on;
% title('The fitting of normalized LTP and LTD curve')

% plot(1:N, LTP_normal_y_exp, 'ro')
% plot(1:N, LTP_normal_y_fit, 'r-')
% plot(1:N, LTP(:,2), 'o', 'MarkerEdgeColor', [1, 0, 0, 0.5], 'MarkerFaceColor', 'none')
% plot(1:N, LTP_normal_y_fit_unnormalize, 'r-', 'LineWidth', 2)

% plot(N+1:2*N, LTD_normal_y_exp, 'bo')
% plot(N+1:2*N, LTD_normal_y_fit, 'b-')
% plot(N+1:2*N, LTD(:,2), 'o', 'MarkerEdgeColor', [0, 1, 0, 0.5], 'MarkerFaceColor', 'none')
% plot(N+1:2*N, LTD_normal_y_fit_unnormalize, 'b-', 'LineWidth', 2)

% legend('LTP','LTP fitting','LTD','LTD fitting','Location', 'northwest')
% xlim([1 128]);
% hold off;

% LTD_normal_y_fit_unnormalize = LTD_normal_y_fit*max_current+starting_LTD(1,2);
% 
% hold on;
% N = 64;
% title('The fitting of normalized LTP and LTD curve')
% plot(1:N, LTP(:,2), 'ro')
% plot(1:N, LTP_normal_y_fit_unnormalize, 'r-')
% plot(N+1:2*N, LTD(:,2), 'bo')
% plot(N+1:2*N, LTD_normal_y_fit_unnormalize, 'b-')
% legend('LTP','LTP fitting','LTD','LTD fitting','Location', 'northwest')
% xlim([1 128]);
% hold off;
