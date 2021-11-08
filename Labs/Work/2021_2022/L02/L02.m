clear all
close all

Data = readmatrix('winequality-red.csv');
X = Data(:,1:11);       % 11 columns for the inputs
X = normalize(X);

N = size(Data,1);       % The number of wines in the set (1599)

Y = Data(:,12) > 5;

% Extend X with 1's
X = [X  ones(N,1)];

% Initialize the 12 params with random values
W = randn(12, 1);   % a column vector

%======================
% Task 1
%======================
z = X * W;
ypred = 1 ./ (1 + exp(-z));
%J = -Y .* log(ypred) - (1 - Y).*log(1-ypred);
%J = mean(J)
eps = 0.0000000001;
for i=1:N
    if (Y(i) == 1)
        J(i) = log(1 / (ypred(i)+eps));
    else
        J(i) = log(  1 / (1-ypred(i) +eps)   );
    end
end
J = mean(J);


%===============================================
% Task 2: train the model with Gradient Descent
%===============================================

number_of_epochs = 100000;    % set number of iterations

for iter = 1:number_of_epochs
    
    % Compute predictions:
    z = X * W;
    ypred = 1 ./ (1 + exp(-z));
    
    % Compute cost:
    eps = 0.0000000001;
    for i=1:N
        if (Y(i) == 1)
            Jeach(i) = log(1 / (ypred(i)+eps));
        else
            Jeach(i) = log(  1 / (1-ypred(i) +eps)   );
        end
    end
    J(iter) = mean(Jeach);
    
    % Compute derivatives according to the given formula
    dW = X' * (ypred - Y);
    
    % Update the weights
    mu = 0.00001;           % try multiple values here
    W = W - mu * dW;
    
    % Store the weights history
    W_hist(:,iter) = W;
end

% Plot the error and the evolution of the weights
plot(J)
%figure
%plot(W_hist)

% Compute final predictions
z = X * W;
ypred = 1 ./ (1 + exp(-z));
ypred = ypred > 0.5;
yvspred = [Y ypred];
        