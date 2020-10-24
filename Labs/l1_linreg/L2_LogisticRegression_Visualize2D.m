clear all
close all

% Load some data
load('LogisticReg.mat');  % the inputs are X, the outputs are Y

% Extend X with a column of 1
N = size(X,1);
X = [X, ones(N,1)];

% Initialize the three parameters in W
W = randn(3,1);

% Repeat Gradient Descent iterations
for iter=1:1000
    % Predict
    z = X * W;
    y_pred = 1./(1 + exp(-z));

    % Cost function
    J(iter) = 1/N * sum(-Y .* log(y_pred) - (1-Y).* log(1-y_pred));
    fprintf('Iteration %d, Error = %g\n', iter, J(iter));

    % Gradient (derivatives)
    dW = X' * (y_pred - Y);

    % Update
    mu = 0.0001;
    W = W - mu*dW;
    
    %===================================
    % Plotting stuff
    %===================================
    
    % Plot decision boundary
    subplot(1,2,1)
    gscatter(X(:,1),X(:,2),Y)
    title('Data plot');
    hold on
    
    % Plot decision line on top of points
    xx = linspace(-2, 3, 1000);
    yy = -W(1)/W(2) * xx - W(3)/W(2);
    hold on
    plot(xx, yy, 'LineWidth',2);
    legend('Class 0', 'Class 1', 'Boundary between classes (output = 0.5)');
    hold off
    axis([-2, 3, -2, 3])
    axis square     
    
    % On right side, plot grayscale regions
    subplot(1,2,2)
    I = zeros(500,500);
    x_values = linspace(-2, 3, 500);
    y_values = linspace(-2, 3, 500);
    for i = 1:length(x_values)
        x = x_values(i);
        for j = 1:length(y_values)
            y = y_values(j);
            % Compute prediction in point (i,j)
            z = [x, y, 1] * W;
            I(501-i,j) = 1 / (1 + exp(-z));   % 0 = black,  1 = white, in-between = gray
        end
    end
    I = fliplr(flipud(I'));
    imshow(I);
    title('Sigmoid output');
    hold on
    
    % Plot line on right side as well
    subplot(1,2,2)
    xx_rescaled = (xx + 2)*500/5+1;
    yy_rescaled = 500 - (yy + 2)*500/5+1;
    %plot(xx_rescaled, yy_rescaled, 'LineWidth',2) 
    plot(xx_rescaled, yy_rescaled, 'LineWidth',2) 
    hold off
    %axis([-2, 3, -2, 3])
    axis square     
    
    drawnow()
   
    %pause(0.1)
    %===================================
end

% clear all
% close all
% 
% load('LogisticReg.mat');
% 
% N = size(X,1);
% X2 = [X, ones(N,1)];
% W = randn(3,1);
% 
% for iter=1:1000
%     % Predict
%     z = X2 * W;
%     y_pred = 1./(1 + exp(-z));
% 
%     % Cost function
%     %J = 1/N * sum((y_pred - Y).^2);
%     J = 1/N * sum(-Y .* log(y_pred) - (1-Y).* log(1-y_pred));
% 
%     % Derivative
%     % dJ/dWi = dJ/dypred * dypred/dz * dz/da
%     %for i = 1:3
%     %    dW(i,1) = 1/N * 2 * sum((y_pred - Y) .* (z .* (1-z)) .* X2(:,i));
%     %end
%     dW = X2' * (y_pred - Y);
% 
%     % Update
%     mu = 0.0001;
%     W = W - mu*dW;
%     
%     dec = 1;
%     if mod(iter,dec) == 0
%         J_store(iter/dec) = J;
%         fprintf('Iter %d, Error = %g\n', iter, J_store(iter/dec));
%         W_store(:,iter/dec) = W;
%     end    
%     
%     % Plot decision boundary
%     subplot(1,2,1)
%     gscatter(X(:,1),X(:,2),Y)
%     hold on
% 
%     % Plot decision line on top of points
%     xx = linspace(-2, 3, 1000);
%     yy = -W(1)/W(2) * xx - W(3)/W(2);
%     hold on
%     plot(xx, yy, 'LineWidth',2);
%     legend('Boundary between classes (output = 0.5)');
%     hold off
%     axis([-2, 3, -2, 3])
%     axis square 
%     
%     % On right side, plot grayscale regions
%     subplot(1,2,2)
%     I = zeros(500,500);
%     x_values = linspace(-2, 3, 500);
%     y_values = linspace(-2, 3, 500);
%     for i = 1:length(x_values)
%         x = x_values(i);
%         for j = 1:length(y_values)
%             y = y_values(j);
%             % Compute prediction in point (i,j)
%             z = [x, y, 1] * W;
%             I(501-i,j) = 1 / (1 + exp(-z));   % 0 = black,  1 = white, in-between = gray
%         end
%     end
%     I = fliplr(flipud(I'));
%     imshow(I);
%     hold on
%     
%     % Plot line on right side as well
%     subplot(1,2,2)
%     xx_rescaled = (xx + 2)*500/5+1;
%     yy_rescaled = 500 - (yy + 2)*500/5+1;
%     %plot(xx_rescaled, yy_rescaled, 'LineWidth',2) 
%     plot(xx_rescaled, yy_rescaled, 'LineWidth',2) 
%     hold off
%     %axis([-2, 3, -2, 3])
%     axis square     
% 
%     drawnow()
%    
%     %pause(0.1)
% end
% 
% % Plot ground truth data
% gscatter(X(:,1),X(:,2),Y)
% 
% % Plot decision line
% xx = linspace(-2, 2, 1000);
% yy = -W(1)/W(2) * xx - W(3);
% hold on
% plot(xx, yy)
% hold off