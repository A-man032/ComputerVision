% Grayscale image denoising running file for the paper:
% "Non-Parametric Bayesian Dictionary Learning for Sparse Image
% Representations," Neural Information Processing Systems (NIPS), 2009.
% Coded by: Mingyuan Zhou, ECE, Duke University, mz1@ee.duke.edu
% Verison 0: 06/14/2009
% Version 1: 09/20/2009
% Version 2: 10/21/2009
% Version 3: 10/28/2009

clear all %清除工作空间的所有变量，函数，和MEX文件
PatchSize = 8; %patch size
K = 256;  %dictionary size
sigma = 25; %noise variance
InitOptions = {'SVD','SVD1','RAND','Kmeans'}; %Initilization options
UpdateOptions = {'DkZkSk','DZS','DZkSk'}; %UpdateOption: In the order of 
%'DkZkSk','DZS', or 'DZkSk'. the computation requirments are 'DkZkSk'<'DZkSk'<'DZS'.


IMname = 'house';
IMin0 = imread([IMname,'.png']); %根据文件名filename读取灰度或彩色图像
%IMin0 = imresize(IMin0,0.25); %改变图像的大小 将图片IMin0放大0.25倍

IMin0=im2double(IMin0)*255; %IMin0 as an input is only used to calculate the 
%PSNR(峰值信噪比) and would not affect the Denoising reults 
%im2double()将灰度图像I转换为双精度，必要时可以缩放其数据。 三维
randn('seed',0) %每次运行产生的随机数都相同 生成符合标准正态的随机数
IMin = fix(min(max((IMin0+sigma*randn(size(IMin0))),0),255)); %噪声矩阵 fix(X) 让X向0靠近取整
%计算PSNR
PSNRIn = 20*log10(255/sqrt(mean((IMin(:)-IMin0(:)).^2))); %mean(A) 当A为矩阵时，那么返回值为该矩阵各列向量的均值

InitOption = InitOptions{1}; UpdateOption = UpdateOptions{1}; IsSubMean = false; 

IterPerRound = ones(PatchSize,PatchSize); %Maximum iteration in each round
IterPerRound(end,end) = 150;

DispPSNR = true; %Calculate and display the PSNR or not
ReduceDictSize = false; %Reduce the ditionary size during training if it is TRUE

%Recommended settings:
%(1) InitOption = 'SVD'; UpdateOption = 'DkZkSk','DZS', or 'DZkSk'; IsSubMean = false;
%(2) InitOption = 'SVD1'; UpdateOption = 'DkZkSk','DZS', or 'DZkSk'; IsSubMean = false; 
%(3) InitOption = 'RAND'; UpdateOption = 'DkZkSk','DZS', or 'DZkSk'; IsSubMean = true; 
%(4) InitOption = 'Kmeans'; UpdateOption = 'DkZkSk','DZS', or 'DZkSk'; IsSubMean = true; 

[ave,Iout,D,S,Z,idex,Pi,NoiseVar,alpha,phi,PSNR,PSNRave] = BPFA_Denoising(IMin, PatchSize, K, InitOption, UpdateOption, IsSubMean, IterPerRound, ReduceDictSize, IMin0,sigma,IMname,DispPSNR);
PSNROut = PSNRave;
figure;
subplot(1,3,1); imshow(IMin0,[]); title('Original clean image');
subplot(1,3,2); imshow(IMin,[]); title((['Noisy image, ',num2str(PSNRIn),'dB']));
subplot(1,3,3); imshow(ave.Iout,[]); title((['Denoised Image, ',num2str(PSNROut),'dB']));
figure;
[temp, Pidex] = sort(Pi,'descend');
Dsort = D(:,Pidex);
I = DispDictionary(Dsort);
title('The dictionary trained on the noisy image');

