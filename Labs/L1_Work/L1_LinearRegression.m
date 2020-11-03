Data = readmatrix('winequality-red.csv');
X = Data(:,1:11);   % 11 columns for the inputs
Y = Data(:,12);     % 1 column for the output
N = size(Data,1);   % The number of wines in the set (1599)

X = [X ones(N,1)];

W = randn(12, 1);   % a column vector

% Task 1 
Yhat = X * W;
J = 1/N * sum((Yhat - Y).^2);

% Task 2
Wopt = (X'*X)^(-1)*X'*Y;
Yhat2 = X * Wopt;
J2 = 1/N * sum((Yhat2 - Y).^2);

% Task 3
mdl = fitlm(X(:,1:11), Y);

% Task 4
W = randn(12, 1);           % initialize parameters randomly
number_of_epochs = 300000;    % set number of iterations

for iter = 1:number_of_epochs
    
    % Compute predictions:
    Ypred = X * W;
    
    % Compute cost:
    J(iter) = 1/N * sum((Ypred - Y).^2);
    
    % Compute derivatives according to the given formula
    dW = X' * (Ypred - Y);
    
    % Update the weights
    mu = 0.0000002;           % try multiple values here
    W = W - mu * dW;
    
    % Store the weights history
    W_hist(:,iter) = W;
end

% Plot the error and the evolution of the weights
plot(J)
%figure
%plot(W_hist)
