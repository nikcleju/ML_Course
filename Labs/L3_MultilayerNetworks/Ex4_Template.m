% MLP Networks with MATLAB on Seeds Dataset

clear all
close all

%% Part 1: Data Preparation

% Load the Seeds dataset
% Replace ... with the actual file path
%===============
% FILL HERE
data = readtable(......); 
%===============

% `data` is a table object, with named columns
% You can access columns like: data.Area, data.Kernel_Length, ...
% Check out all the available columns

% Data Exploration
% Create a scatter plot between two features (example: area vs perimeter)
% scatter(data.Area, data.Perimeter);
xlabel('Area'); ylabel('Perimeter');
title('Area vs Perimeter');

% TODO: Create a scatter plot between kernel length and kernel width
figure   % new figure
%===============
% FILL HERE
......
%===============

% Split input and output
features = data(:, 1:end-1);   % inputs
labels = data(:, end);         % outputs, assuming its the last column

% Convert all data from table to array
features   = table2array(features);
labels     = table2array(labels);

% One-hot encoding of labels
labels_onehot = onehotencode(categorical(labels),2);

% Data normalization using `normalization()`
%===============
% FILL HERE
features = ......
%===============

%% Part 2: Building and Training the Network

% Define the network architecture using `patternnet()`
%===============
% FILL HERE
net = ......
%===============

% Set up training parameters (optional)
net.trainParam.epochs = 100;                % Max number of epochs

% Train the network
%===============
% FILL HERE
[net, tr] = train( ......  );               
%===============
