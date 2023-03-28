clc; clear; 
close all;
%% Önce dataları okuyoruz-
dataSetDir = fullfile('D:\state\train1');
imds = imageDatastore(dataSetDir,'IncludeSubfolders',1,'LabelSource','foldernames');
%% Datayı train ve validation olarak bölüyoruz
[imdsTrain1,imdsTest] = splitEachLabel(imds,0.8,'randomize');
[imdsTrain,imdsValidation] = splitEachLabel(imdsTrain1,0.9,'randomized');
%% Ağ parametrelerini belirliyoruz
kanalBoyutu=3;
imageSize = [480 640 3];
numClasses = 5; 
%% Ağ yapısını oluşturuyoruz
load('ya_graph.mat');

lgraph = ya_graph;
% %%
% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.01,...
%     'LearnRateDropFactor',0.9,...
%     'LearnRateDropPeriod',1,...
%     'LearnRateSchedule','piecewise',....
%     'MaxEpochs',5, ...
%     'Shuffle','every-epoch', ...
%     'Verbose',true, ...
%     'MiniBatchSize',4, ...
%     'ValidationData',imdsValidation, ...
%     'ValidationFrequency',1000, ...
%     'Plots','training-progress');
% 
% % Oluşturulan Ağ
% net = trainNetwork(imdsTrain,ya_graph,options);
% 
%
load('net.mat');

YPred = classify(net,imdsTest);
YValidation = imdsTest.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);




