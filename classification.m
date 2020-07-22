clc;
close all;
clear;

net = alexnet;


layers = net.Layers(1:end-3);
new_layers = [layers
              fullyConnectedLayer(10, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
              softmaxLayer
              classificationLayer];

image = imageDatastore('Pictures_Resize',...
    'IncludeSubfolders', true,...
    'LabelSource', 'foldernames');


[imageTrain, imageTest] = splitEachLabel(image, 0.8, 'randomized');

%%%%%%%%%%%%%%%%%%%%%%% 下面的两行是新加入的，可以用但是太浪费时间了 %%%%%%%%%%%%%%%%%%%%%%%
%imageTrain = augmentedImageDatastore([227 227], imageTrain);   % 输入自动变成227*227
%imageTest = augmentedImageDatastore([227 227], imageTest);     % 输入自动变成227*227



ops = trainingOptions('sgdm', ...
                      'InitialLearnRate',0.0001, ...
                      'ValidationData',imageTest, ...
                      'Plots','training-progress', ...
                      'MiniBatchSize',5, ...
                      'MaxEpochs',10,...
                      'ValidationPatience',Inf,...
                      'Verbose',false);
 
% 'MaxEpochs' 即训练次数，根据需要调整


%开始训练
tic
net_train = trainNetwork(imageTrain, new_layers, ops);
toc

% 将 net_train 保存为 my_net
save my_net net_train



