%
net = alexnet('Weights','imagenet');
%net = resnet18('Weights','imagenet');
analyzeNetwork(net)  % or double-click `net` in the Workspace

%I = imread('OIP.jpg');
%I = imread('peppers.png');
I = imread('Pen.jpg');
I = double(I);
I = imresize(I, [227 227]);

imshow(I/255)

[YPred, scores] = classify(net,I)
disp(YPred)
%[maxval, maxpos] = max(scores)