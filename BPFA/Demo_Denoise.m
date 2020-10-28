% Grayscale image denoising running file for the paper:
% "Non-Parametric Bayesian Dictionary Learning for Sparse Image
% Representations," Neural Information Processing Systems (NIPS), 2009.
% Coded by: Mingyuan Zhou, ECE, Duke University, mz1@ee.duke.edu
% Verison 0: 06/14/2009
% Version 1: 09/20/2009
% Version 2: 10/21/2009
% Version 3: 10/28/2009

clear all
PatchSize = 8; %patch size
K = 256;  %dictionary size
sigma = 25; %noise variance
InitOptions = {'SVD','SVD1','RAND','Kmeans'}; %Initilization options
UpdateOptions = {'DkZkSk','DZS','DZkSk'}; %UpdateOption: In the order of 
%'DkZkSk','DZS', or 'DZkSk'. the computation requirments are 'DkZkSk'<'DZkSk'<'DZS'.


IMname = 'house';
IMin0 = imread([IMname,'.png']);
%IMin0 = imresize(IMin0,0.25);

IMin0=im2double(IMin0)*255; %IMin0 as an input is only used to calculate the 
%PSNR and would not affect the Denoising reults 
randn('seed',0)
IMin = fix(min(max((IMin0+sigma*randn(size(IMin0))),0),255));
PSNRIn = 20*log10(255/sqrt(mean((IMin(:)-IMin0(:)).^2)));

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

[ave,Iout,D,S,Z,idex,Pi,NoiseVar,alpha,phi,PSNR,PSNRave] = BPFA_Denoise(IMin, PatchSize, K, InitOption, UpdateOption, IsSubMean, IterPerRound, ReduceDictSize, IMin0,sigma,IMname,DispPSNR);
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

