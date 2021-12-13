%=======================================
% Just for using the pretrained model
%=======================================
doTraining = false;
if ~doTraining && ~exist('ssdResNet50VehicleExample_20a.mat','file')
    disp('Downloading pretrained detector (44 MB)...');
    pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/ssdResNet50VehicleExample_20a.mat';
    websave('ssdResNet50VehicleExample_20a.mat',pretrainedURL);
end

if doTraining
    % Train the SSD detector.
    [detector, info] = trainSSDObjectDetector(preprocessedTrainingData,lgraph,options);
else
    % Load pretrained detector for the example.
    pretrained = load('ssdResNet50VehicleExample_20a.mat');
    detector = pretrained.detector;
end

% Load image
I = imread('Image1.jpg');
%I = imread('vehicleImages/image_00002.jpg');
I = double(I) / 255;
[bboxes,scores] = detect(detector,I);  %, 'Threshold', 0.4
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)