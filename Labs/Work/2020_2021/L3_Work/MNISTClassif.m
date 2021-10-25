clear all
close all

load mnist_train.csv

X = mnist_train(:,2:end);
Y = mnist_train(:,1);
Yenc = full(ind2vec(1 + Y')) % digits are 0 to 9, add 1 so that class indices are 1 to 10 (10 classes)
Yenc = Yenc';          % Transpose, so each value corresponds to a row, just like the input data

% 
mygreatnetwork = patternnet([80, 60]);            % Use two hidden layers, with size 80 and 60
mygreatnetwork = train(mygreatnetwork, X', Yenc');

% Test network
I = imread('MyImage.png');
I = double(rgb2gray(I));
% Convert to line
I = I';
Ivec = I(:);
Ivec = Ivec';

% Do inference
outputs = mygreatnetwork(Ivec);
