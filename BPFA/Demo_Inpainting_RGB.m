% RGB image inpainting running file for the paper:
% "Non-Parametric Bayesian Dictionary Learning for Sparse Image
% Representations," Neural Information Processing Systems (NIPS), 2009.
% Coded by: Mingyuan Zhou, ECE, Duke University, mz1@ee.duke.edu
% Version 0: 06/14/2009
% Version 1: 09/21/2009
% Version 2: 10/26/2009
% Version 3: 10/28/2009

clear all
PatchSize = 8; %block size
K = 256;  %dictionary size
InitOptions = {'SVD','SVD1','RAND','Kmeans'}; %Initilization options
UpdateOptions = {'DkZkSk','DZS','DZkSk'}; % Update options, the computation requirments are 'DkZkSk'<'DZkSk'<'DZS'

IMname = 'Castle';
%IMname = 'NewOrlean';
if strcmp(IMname,'Castle')==1
    IMin0 = imread('castle_original.png');
    IMin0 = im2double(IMin0);
    IMin0 = IMin0*255;  %IMin0 as an input is only used to calculate the  
    %PSNR and would not affect the Inpainting reults
    IMin = imread('castle_damaged.png');
    IMin = im2double(IMin);
    IMin = IMin*255;
    SampleMatrix = abs(IMin-IMin0)<1;  %binary matrix indicating which pixel values are observed  
    IMin = IMin0.*SampleMatrix;   
    %recommended setting 1:
    InitOption = InitOptions{1}; UpdateOption = UpdateOptions{1}; IsSubMean = false; 
    %recommended setting 2:
    %InitOption = InitOptions{2}; UpdateOption = UpdateOptions{1}; IsSubMean = false; 
elseif strcmp(IMname,'NewOrlean')==1
    IMin0 = imread('fig13_original.png');
    IMin0=im2double(IMin0);
    IMin0 = IMin0*255;  %IMin0 as an input is only used to calculate the  
    %PSNR and would not affect the Inpainting reults
    IMin = imread('fig13_damaged.bmp');
    IMin=im2double(IMin);
    IMin = IMin*255;
    SampleMatrix = false(size(IMin));
    SampleMatrix(:,:,1) = IMin(:,:,1)<252;  
    SampleMatrix(:,:,2) = SampleMatrix(:,:,1);
    SampleMatrix(:,:,3) = SampleMatrix(:,:,1);    %binary matrix indicating which pixel values are observed
    IMin = IMin0.*SampleMatrix;
    %recommended setting 1:
    InitOption = InitOptions{2}; UpdateOption = UpdateOptions{1}; IsSubMean = false; 
    %recommended setting 2:
    %InitOption = InitOptions{1}; UpdateOption = UpdateOptions{2}; IsSubMean = false; 
end
PSNRIn = 20*log10(255/sqrt(mean((IMin(:)-IMin0(:)).^2)));

IterPerRound = ones(PatchSize,PatchSize); %Maximum iteration in each round
IterPerRound(end,end) = 50;

DispPSNR = true; %Calculate and display the PSNR or not
ReduceDictSize = false; %Reduce the ditionary size during training if it is TRUE
SparseCalculationT = 0.6; %A threshold used to decide whether sparse 
%calculations should be used based on the percentage of the observed data, 
%it would only affect the speed and has no effect on the performance.

%Recommended settings:
%(1) InitOption = 'SVD'; UpdateOption = 'DkZkSk','DZS', or 'DZkSk'; IsSubMean = false;
%(2) InitOption = 'SVD1'; UpdateOption = 'DkZkSk','DZS', or 'DZkSk'; IsSubMean = false; 
%(3) InitOption = 'RAND'; UpdateOption = 'DkZkSk','DZS', or 'DZkSk'; IsSubMean = true; 
%(4) InitOption = 'Kmeans'; UpdateOption = 'DkZkSk','DZS', or 'DZkSk'; IsSubMean = true; 

[Iout,D,S,Z,idex,Pi,NoiseVar,alpha,phi,PSNR] = BPFA_Inpainting(IMin, SampleMatrix, PatchSize, K, InitOption, UpdateOption, IsSubMean, IterPerRound, ReduceDictSize,IMin0,IMname,DispPSNR,SparseCalculationT);
PSNROut = PSNR(end);
figure;
subplot(1,3,1); imshow(IMin/255); title(['Corrupted image, ',num2str(PSNRIn),'dB']);
subplot(1,3,2); imshow(IMin0/255); title('Original image');
subplot(1,3,3); imshow(Iout/255); title(['Restored image, ',num2str(PSNROut),'dB']);
figure;
[temp, Pidex] = sort(Pi,'descend');
Dsort = D(:,Pidex);
I = DispDictionary(Dsort,'rgb'); %Display the dictionary elements as a image
title('The dictionary trained on the corrupted image');