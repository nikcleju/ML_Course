%clear all
%close all

%load('mnist_train.csv')

X = mnist_train(:, 2:end);
Y = mnist_train(:, 1);

% Let's plot one image
row = X(3423,:)/255;
imshow(reshape(row, 28, 28)')

% One hot encoding
Yenc = full(ind2vec(1 + Y'));   % digits are 0 to 9, add 1 so that class indices are 1 to 10 (10 classes)
Yenc = Yenc';

% nprtool - graphical tool

mygreatnetwork = patternnet([80, 60]);
mygreatnetwork = train(mygreatnetwork, X',Yenc');  % Train the network with our data. The data should be arranged properly

% Test
v = X(1,:);
mygreatnetwork(v')

% Test with my own image
I = imread('Mine.png');
I = double(rgb2gray(I));
imshow(I)
Ivec = I(:);
mygreatnetwork(Ivec)
