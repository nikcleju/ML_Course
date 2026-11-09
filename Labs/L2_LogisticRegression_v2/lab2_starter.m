%% 0. Setup
% Run sections in order. Complete TODO A-D, then write code in sections 5-6.
clear; close all; clc;
labFolder = fileparts(mfilename('fullpath'));
assert(~isempty(labFolder), 'Save and run the script, not pasted commands.');
assert(exist('fitglm', 'file') ~= 0, ...
    'This lab needs Statistics and Machine Learning Toolbox.');

%% 1. Synthetic example: probabilities and a boundary
S = readmatrix(fullfile(labFolder, 'data', 'synthetic_points.csv'));
Xdemo = S(:,1:2);
ydemo = S(:,3);
mdlDemo = fitglm(Xdemo, ydemo, 'Distribution', 'binomial');
coeffDemo = mdlDemo.Coefficients.Estimate;
fprintf('Synthetic bias, weight 1, weight 2:\n');
disp(coeffDemo);

examplePoint = [0 -0.3];
pExample = predict(mdlDemo, examplePoint);
classExample = pExample >= 0.5;
fprintf('Point (0, -0.3): p(class 1) = %.3f; predicted class = %d\n', ...
    pExample, classExample);

% This grid is for plotting only. It is not additional training data.
x1 = linspace(min(Xdemo(:,1))-0.5, max(Xdemo(:,1))+0.5, 120);
x2 = linspace(min(Xdemo(:,2))-0.5, max(Xdemo(:,2))+0.5, 120);
[G1, G2] = meshgrid(x1, x2);
Pgrid = reshape(predict(mdlDemo, [G1(:), G2(:)]), size(G1));

figure(1);
contourf(G1, G2, Pgrid, 20, 'LineStyle', 'none');
caxis([0 1]); colorbar; hold on;
h0 = scatter(Xdemo(ydemo==0,1), Xdemo(ydemo==0,2), 30, 'k', 'o', 'filled');
h1 = scatter(Xdemo(ydemo==1,1), Xdemo(ydemo==1,2), 35, 'w', '^', 'filled');
[~, hBoundary] = contour(G1, G2, Pgrid, [0.5 0.5], 'r-', 'LineWidth', 2);
hold off;
xlabel('x_1 (artificial units)'); ylabel('x_2 (artificial units)');
title('Colour = estimated P(class 1); red line = threshold 0.5');
legend([h0 h1 hBoundary], 'Class 0', 'Class 1', 'Decision boundary', ...
    'Location', 'best');

%% 2. Inspect Titanic data and establish a baseline
D = readmatrix(fullfile(labFolder, 'data', 'titanic_lab.csv'));
% Columns: sex code, transformed age, transformed fare, survived, split flag, ID.
X = D(:,1:3);
yAll = D(:,4);
trainRows = D(:,5) == 1;
testRows = D(:,5) == 0;
Xtrain = X(trainRows,:);
Xtest = X(testRows,:);
ytrain = yAll(trainRows);
ytest = yAll(testRows);
assert(isequal(size(Xtrain), [791 3]));
assert(isequal(size(Xtest), [100 3]));
assert(isempty(intersect(D(trainRows,6), D(testRows,6))), ...
    'Training and test passenger IDs must be disjoint.');
disp('First five rows (see worksheet for column meanings):');
disp(D(1:5,:));
disp('Training and test predictor-matrix sizes:');
disp(size(Xtrain)); disp(size(Xtest));
fprintf('Training counts: class 0 = %d, class 1 = %d\n', ...
    sum(ytrain==0), sum(ytrain==1));
fprintf('Test counts: class 0 = %d, class 1 = %d\n', ...
    sum(ytest==0), sum(ytest==1));

baselineClass = double(mean(ytrain) >= 0.5); % Ties go to class 1.
baselinePred = baselineClass * ones(size(ytest));
accuracyBaseline = mean(baselinePred == ytest);
fprintf('Baseline predicts class %d; test accuracy = %.1f%%\n', ...
    baselineClass, 100*accuracyBaseline);

%% 3. One-predictor model: sex code
% TODO A: Select column 1 of Xtrain.
xSingle = [];
assert(isequal(size(xSingle), [791 1]), ...
    'TODO A: select the training sex-code column (791-by-1).');

mdlSingle = fitglm(xSingle, ytrain, 'Distribution', 'binomial');
pSingle = predict(mdlSingle, Xtest(:,1));

% TODO B: Convert pSingle to class labels using threshold 0.5.
classSingle = [];
assert(isequal(size(classSingle), size(ytest)) && ...
    all(classSingle==0 | classSingle==1), ...
    'TODO B: produce one 0/1 class per test passenger.');

% TODO C: Compute the fraction of correct test class decisions.
accuracySingle = NaN;
assert(isscalar(accuracySingle) && isfinite(accuracySingle) && ...
    accuracySingle >= 0 && accuracySingle <= 1, ...
    'TODO C: compute an accuracy fraction between 0 and 1.');

fprintf('One-predictor bias and weight:\n');
disp(mdlSingle.Coefficients.Estimate);
fprintf('One-predictor test accuracy = %.1f%%\n', 100*accuracySingle);

%% 4. Three-predictor model: sex code, transformed age, transformed fare
% TODO D: Fit logistic regression using Xtrain and ytrain.
mdlMultiple = [];
assert(~isempty(mdlMultiple), 'TODO D: fit the three-predictor model.');

pMultiple = predict(mdlMultiple, Xtest);
classMultiple = pMultiple >= 0.5;
accuracyMultiple = mean(classMultiple == ytest);
fprintf('Three-predictor bias, sex-code, age, fare weights:\n');
disp(mdlMultiple.Coefficients.Estimate);

% Training cost, not test accuracy, is the quantity minimized in fitting.
% Clipping avoids log(0) if floating-point probabilities round to 0 or 1.
pTrainSingle = min(max(predict(mdlSingle, xSingle), eps), 1-eps);
pTrainMultiple = min(max(predict(mdlMultiple, Xtrain), eps), 1-eps);
costSingle = -mean(ytrain.*log(pTrainSingle) + ...
    (1-ytrain).*log(1-pTrainSingle));
costMultiple = -mean(ytrain.*log(pTrainMultiple) + ...
    (1-ytrain).*log(1-pTrainMultiple));
fprintf('\nModel                  Training cross-entropy    Test accuracy\n');
fprintf('Majority-class baseline          --                 %5.1f%%\n', 100*accuracyBaseline);
fprintf('Sex code only                  %.4f                 %5.1f%%\n', costSingle, 100*accuracySingle);
fprintf('Sex code, age, fare             %.4f                 %5.1f%%\n', costMultiple, 100*accuracyMultiple);

% Rows = actual class [0,1]; columns = predicted class [0,1].
C = [sum(ytest==0 & classMultiple==0), sum(ytest==0 & classMultiple==1); ...
     sum(ytest==1 & classMultiple==0), sum(ytest==1 & classMultiple==1)];
assert(sum(C(:)) == numel(ytest));
disp('Confusion matrix: actual rows 0,1; predicted columns 0,1:');
disp(C);
figure(2);
imagesc(C); colorbar;
xticks([1 2]); xticklabels({'0', '1'});
yticks([1 2]); yticklabels({'0', '1'});
xlabel('Predicted class'); ylabel('Actual class');
title('Three-predictor model: 100 held-out passengers');
for r = 1:2
    for c = 1:2
        text(c, r, num2str(C(r,c)), 'HorizontalAlignment', 'center', ...
            'FontWeight', 'bold', 'Color', 'k', 'BackgroundColor', 'w');
    end
end

%% 5. Calculate a prediction yourself (three lines)
xManual = Xtest(1,:); % A row containing three predictor values.
coeffManual = mdlMultiple.Coefficients.Estimate;
bManual = coeffManual(1);
wManual = coeffManual(2:end); % A column containing three weights.

% Write your code here: zManual, pManual, classManual.
% Use the model equation and sigmoid, not predict().

assert(exist('zManual', 'var') == 1 && exist('pManual', 'var') == 1 && ...
    exist('classManual', 'var') == 1, ...
    'Task 5: write three lines computing zManual, pManual, classManual.');
pLibrary = predict(mdlMultiple, xManual);
assert(isscalar(pManual) && isfinite(pManual) && abs(pManual-pLibrary) < 1e-10, ...
    'Task 5: check the linear score and sigmoid calculation.');
assert(isscalar(classManual) && classManual == (pManual >= 0.5), ...
    'Task 5: use threshold 0.5 for the class decision.');
fprintf('Manual probability = %.6f; predict() = %.6f; class = %d\n', ...
    pManual, pLibrary, classManual);

%% 6. New dataset: write the fit-predict-evaluate sequence (about six lines)
B = readmatrix(fullfile(labFolder, 'data', 'spine_lab.csv'));
% Columns: 12 measurements, abnormal (0/1), split flag, source row.
spineTrainRows = B(:,14) == 1;
spineTestRows = B(:,14) == 0;
XspineTrain = B(spineTrainRows,1:12);
XspineTest = B(spineTestRows,1:12);
yspineTrain = B(spineTrainRows,13);
yspineTest = B(spineTestRows,13);
assert(isequal(size(XspineTrain), [248 12]));
assert(isequal(size(XspineTest), [62 12]));
assert(isempty(intersect(B(spineTrainRows,15), B(spineTestRows,15))));
spineBaselineClass = double(mean(yspineTrain) >= 0.5);
spineBaselineAccuracy = mean(yspineTest == spineBaselineClass);

% Write your code here (see Task 6):
% mdlSpine, pSpine, classSpine, accuracySpine, falseNegSpine, falsePosSpine.

assert(exist('mdlSpine', 'var') == 1 && exist('pSpine', 'var') == 1 && ...
    exist('classSpine', 'var') == 1 && exist('accuracySpine', 'var') == 1 && ...
    exist('falseNegSpine', 'var') == 1 && exist('falsePosSpine', 'var') == 1, ...
    'Task 6: write the six fit-predict-evaluate lines before running this section.');
assert(isequal(size(pSpine), [62 1]) && all(isfinite(pSpine)) && ...
    all(pSpine >= 0 & pSpine <= 1), 'Task 6: predict 62 test probabilities.');
assert(isequal(size(classSpine), [62 1]) && all(classSpine==0 | classSpine==1), ...
    'Task 6: use threshold 0.5 to produce 62 test class decisions.');
assert(isscalar(accuracySpine) && isfinite(accuracySpine) && ...
    accuracySpine >= 0 && accuracySpine <= 1, ...
    'Task 6: compute the fraction of correct test predictions.');
assert(isscalar(falseNegSpine) && isscalar(falsePosSpine) && ...
    isfinite(falseNegSpine) && isfinite(falsePosSpine) && ...
    falseNegSpine >= 0 && falsePosSpine >= 0 && ...
    falseNegSpine == fix(falseNegSpine) && falsePosSpine == fix(falsePosSpine) && ...
    abs(falseNegSpine + falsePosSpine - 62*(1-accuracySpine)) < 1e-10, ...
    'Task 6: error counts must be nonnegative integers consistent with accuracy.');
fprintf('Spine majority baseline = %.1f%%; model test accuracy = %.1f%%\n', ...
    100*spineBaselineAccuracy, 100*accuracySpine);
fprintf('Spine: false negatives = %d; false positives = %d\n', ...
    falseNegSpine, falsePosSpine);

% Optional squared-feature comparison: write your additional code here.

%% 7. Self-check and screenshot report (see worksheet)
