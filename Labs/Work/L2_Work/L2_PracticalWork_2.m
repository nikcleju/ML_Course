clear all
close all

Data = readmatrix('winequality-red.csv');
X = Data(:, 1:11);       % 11 columns for the inputs
N = size(Data,1);       % The number of wines in the set (1599)

Y = Data(:,12) > 5;     % make 1 column for the output: 1 if score > 5, 0 if score <= 5

X = [X  ones(N, 1)];

%======================
% To fill in
%======================

W = randn(12, 1);           % initialize parameters randomly

number_of_epochs = 1000;    % set number of iterations

for iter = 1:number_of_epochs
    
    % Compute predictions:
    z = X * W;
    Y_pred = 1 / (1 + exp(-z));
    Y_pred = Y_pred';
    
    % Compute cost:
    J(iter) = 1/N * sum(   -Y .* log(Y_pred + 0.00000000001) - (1-Y).*log(1-Y_pred + 0.00000000001)    );
    
    % Compute derivatives according to the given formula
    dW = X' * (Y_pred - Y);
    
    % Update the weights
    mu = 0.000001;           % try multiple values here
    W = W - mu * dW;
    
    % Store the weights history
    W_hist(:,iter) = W;
end

% Plot the error and the evolution of the weights
plot(J)
figure
plot(W_hist)