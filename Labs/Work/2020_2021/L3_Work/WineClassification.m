
% Read dataset
Data = readmatrix('winequality-red.csv');
X = Data(:,1:11);       % 11 columns for the inputs
N = size(Data,1);       % The number of wines in the set (1599)
Y = Data(:,12) ;        % 12th column the quality

Yenc = full(ind2vec(Y'));
Yenc = Yenc(3:end, :); % Remove first two lines
Yenc = Yenc';          % Transpose, so each value corresponds to a row, just like the input data

nprtool()