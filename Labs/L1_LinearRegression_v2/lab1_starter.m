%% 0. Setup
% Run sections in order. Keep the data subfolder beside this saved script.
clear; close all; clc;

labFolder = fileparts(mfilename('fullpath'));
assert(~isempty(labFolder), 'Save and run the script, not pasted commands.');
assert(exist('fitlm', 'file') ~= 0, ...
    'This lab needs MATLAB with Statistics and Machine Learning Toolbox.');

%% 1. Fit a noisy straight line
S = readmatrix(fullfile(labFolder, 'data', 'synthetic_line.csv'));
x = S(:,1);
y = S(:,2);

bCandidate = 0;
wCandidate = 1;
yCandidate = bCandidate + wCandidate*x;
Jcandidate = mean((yCandidate-y).^2);

mdlDemo = fitlm(x, y);  % Minimizes training cost; fits a bias.
yDemoTrain = predict(mdlDemo, x);
Jtrained = mean((yDemoTrain-y).^2);

bDemo = mdlDemo.Coefficients.Estimate(1);
wDemo = mdlDemo.Coefficients.Estimate(2);

xLine = linspace(min(x), max(x), 100)';
yLine = predict(mdlDemo, xLine);

figure(1);
scatter(x, y, 25, 'filled'); hold on;
plot(xLine, bCandidate+wCandidate*xLine, 'k--', 'LineWidth', 1.5);
plot(xLine, yLine, 'r-', 'LineWidth', 2); hold off;
xlabel('x (artificial units)'); ylabel('y (artificial units)');
legend('Observations', 'Untrained candidate', 'Trained line', 'Location', 'best'); grid on;

fprintf('Synthetic: bias = %.3f, weight = %.3f\n', bDemo, wDemo);
fprintf('Training MSE cost: candidate = %.3f, trained = %.3f\n', Jcandidate, Jtrained);

%% 2. Load and inspect the car data
D = readmatrix(fullfile(labFolder, 'data', 'auto_mpg_lab.csv'));
% Columns: vehicle weight, displacement, acceleration time, MPG, split flag, row ID.

X = D(:,1:3);
yAll = D(:,4);

trainRows = D(:,5) == 1;
testRows = D(:,5) == 0;

Xtrain = X(trainRows,:);
Xtest = X(testRows,:);
ytrain = yAll(trainRows);
ytest = yAll(testRows);

assert(isequal(size(Xtrain), [318 3]));
assert(isequal(size(Xtest), [80 3]));

fprintf('Training: %d rows; test: %d rows.\n', numel(ytrain), numel(ytest));

% ======================================================================
% STUDENT CODE A -- replace the placeholder on the line below
% TODO A: Select the vehicle-weight column of Xtrain.
% ----------------------------------------------------------------------
xWeight = [];
% END STUDENT CODE A
% ======================================================================

assert(isequal(size(xWeight), [318 1]), ...
    'TODO A: xWeight must be the 318-by-1 training vehicle-weight column.');

figure(2);
scatter(xWeight, ytrain, 25, 'filled');
xlabel('Vehicle weight (1000 lb)'); ylabel('Fuel economy (mpg)'); grid on;

%% 3. Training-mean baseline
baselineValue = mean(ytrain);
baselinePred = baselineValue * ones(size(ytest));
baselineRMSE = sqrt(mean((baselinePred - ytest).^2));
fprintf('Baseline test RMSE = %.3f mpg\n', baselineRMSE);

%% 4. Model using vehicle weight only
% ======================================================================
% STUDENT CODE B -- replace the placeholder on the line below
% TODO B: Select the vehicle-weight column of Xtest.
% ----------------------------------------------------------------------
xWeightTest = [];
% END STUDENT CODE B
% ======================================================================

assert(isequal(size(xWeightTest), [80 1]), ...
    'TODO B: xWeightTest must be the 80-by-1 test vehicle-weight column.');

mdlSingle = fitlm(xWeight, ytrain);
predSingle = predict(mdlSingle, xWeightTest);

% ======================================================================
% STUDENT CODE C -- replace the placeholder on the line below
% TODO C: Compute test RMSE from predSingle and ytest (use the baseline as a pattern).
% ----------------------------------------------------------------------
rmseSingle = NaN;
% END STUDENT CODE C
% ======================================================================

assert(isscalar(rmseSingle) && isfinite(rmseSingle), ...
    'TODO C: compute one finite scalar RMSE.');

bSingle = mdlSingle.Coefficients.Estimate(1);
wSingle = mdlSingle.Coefficients.Estimate(2);
fprintf('Vehicle-weight model: bias = %.3f, weight = %.3f\n', bSingle, wSingle);
fprintf('Vehicle-weight model test RMSE = %.3f mpg\n', rmseSingle);

xGrid = linspace(min(xWeight), max(xWeight), 100)';
figure(2); hold on;
plot(xGrid, predict(mdlSingle, xGrid), 'r-', 'LineWidth', 2); hold off;
legend('Training observations', 'Training-fitted line', 'Location', 'best');

%% 5. Three-predictor model
% ======================================================================
% STUDENT CODE D -- replace the placeholder on the line below
% TODO D: Fit a model using all three predictors and ytrain.
% ----------------------------------------------------------------------
mdlMultiple = [];
% END STUDENT CODE D
% ======================================================================

assert(~isempty(mdlMultiple), 'TODO D: fit the three-predictor model.');

predMultiple = predict(mdlMultiple, Xtest);
rmseMultiple = sqrt(mean((predMultiple - ytest).^2));

trainPredSingle = predict(mdlSingle, xWeight);
trainPredMultiple = predict(mdlMultiple, Xtrain);
trainCostSingle = mean((trainPredSingle - ytrain).^2);
trainCostMultiple = mean((trainPredMultiple - ytrain).^2);

fprintf('\nModel                  Training MSE      Test RMSE (mpg)\n');
fprintf('Training-mean baseline        --          %8.3f\n', baselineRMSE);
fprintf('Vehicle weight only       %8.3f          %8.3f\n', trainCostSingle, rmseSingle);
fprintf('Three predictors          %8.3f          %8.3f\n', trainCostMultiple, rmseMultiple);

%% 6. Discussion and self-check (see worksheet)
