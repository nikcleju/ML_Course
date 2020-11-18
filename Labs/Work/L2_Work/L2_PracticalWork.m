Data = readmatrix('winequality-red.csv');
X = Data(:, 1:11);       % 11 columns for the inputs

N = size(Data,1);       % The number of wines in the set (1599)

Y = Data(:,12) > 5;     % make 1 column for the output: 1 if score > 5, 0 if score <= 5

% New wine
newwine = [8	0.33	0.53	2.5	0.091	18	80	0.9976	3.37	0.8	9.6];

% Method 1: with Matlab's fitlm() function
mdl1 = fitglm(X, Y, 'Distribution', 'binomial')

% Predict 
Yhat = mdl1.predict(X);