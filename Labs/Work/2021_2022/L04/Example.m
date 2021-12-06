%clear all
%close all

% Explore AlexNet network
%net = alexnet('Weights','imagenet')
%analyzeNetwork(net)

% Classify an image with AlexNet
%I  = imread('HotDog.jpg');
%I  = imread('Umbrella.jpg');
%I  = imread('Umbrella2.jpg');
I  = imread('Cat.jpg');
imshow(I)
I2 = imresize(I, [227 227]);
figure
imshow(I2)
net.classify(I2)
p = net.predict(I2);
figure
plot(p)
