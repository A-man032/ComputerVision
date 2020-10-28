function [Iout,D,S,Z,idex,Pi,NoiseVar,alpha,phi,PSNR] = BPFA_Inpainting(IMin, SampleMatrix, PatchSize, K, InitOption, UpdateOption, IsSubMean, IterPerRound, ReduceDictSize,IMin0,IMname,DispPSNR,SparseCalculationT)
%------------------------------------------------------------------
% The BPFA RGB image inpainting program for the following paper:
% "Non-Parametric Bayesian Dictionary Learning for Sparse Image
% Representations," Neural Information Processing Systems (NIPS), 2009.
% Coded by: Mingyuan Zhou, ECE, Duke University, mz1@ee.duke.edu
% Version 0: 06/14/2009
% Version 1: 09/13/2009
% Version 2: 10/26/2009
% Version 3: 10/28/2009
% Version 4: 10/29/2009
%------------------------------------------------------------------
% Input:
%   IMin: corrupted image
%   SampleMatrix: binary matrix indicating which pixel values are observed
%   PatchSize: block size, 8 is commonly used.
%   K: predefined dictionary size.
%   InitOption: SVD,SVD1,RAND,Kmeans.
%   UpdateOption: In the order of 'DkZkSk','DZS', or 'DZkSk'. the
%   computation requirments are 'DkZkSk'<'DZkSk'<'DZS'.
%   IsSubMean: substract the globle mean if it is TRUE.
%   IterPerRound: maximum iteration in each round.
%   ReduceDictSize: reduce the dictionary size during learning if it is
%   TRUE.
%   IMin0: original noise-free image (only used for PSNR calculation and
%   would not affect the denoising results).
%   IMname: image name.
%   DispPSNR: calculate and display the instant PSNR during learning if it
%   is TRUE
% Output:
%   ave: inpainted image (averaged output)
%   Iout: inpainted image (sample output)
%   D: dictionary learned from the corrupted image
%   S: basis coefficients
%   Z: binary indicators
%   idex: patch index
%   Pi: the probabilities for each dictionary entries to be used
%   NoiseVar: estimated noise variance
%   alpha: precision for S
%   phi: noise precision
%   PSNR: peak signal-to-noise ratio
%   PSNRave: PSNR of ave.Iout
%------------------------------------------------------------------
if nargin<3
    PatchSize=8;
end
if nargin < 4
    K=256;
end
if nargin <5
    InitOption='SVD';
end
if nargin <6
    UpdateOption = 'DkZkSk';
end
if nargin <7
    IsSubMean=false;
end
if nargin < 8
    IterPerRound = ones(PatchSize,PatchSize);
    IterPerRound(1,1) = 10;
    IterPerRound(1,2:end) = 2;
    IterPerRound(end,end) = 10;
end
if nargin < 9
    ReduceDictSize = false;
end
if nargin < 10
    IMin0=IMin;
end
if nargin <11
    IMname=[];
end
if nargin <12
    DispPSNR = false;
end
if nargin < 13
    SparseCalculationT = 0.5;
end

if size(IMin,3)==1 %图片第三维尺寸为1即为灰度图片
    ImageType='gray';
elseif size(IMin,3)==3 %图片第三维尺寸为1即为彩色图片
    ImageType='rgb';
else
    ImageType='hyperspec';
end


idex=[];
PSNR=[];
NoiseVar=[];

if strcmp(ImageType, 'rgb')==1
    MuX1 = zeros(3*PatchSize^2,1);
    MuX1(1:PatchSize^2) = sum(sum(IMin(:,:,1)/255))/nnz(SampleMatrix(:,:,1));
    MuX1(PatchSize^2+(1:PatchSize^2)) = sum(sum(IMin(:,:,2)/255))/nnz(SampleMatrix(:,:,2));
    MuX1(2*PatchSize^2+(1:PatchSize^2)) = sum(sum(IMin(:,:,3)/255))/nnz(SampleMatrix(:,:,3));
    if IsSubMean
        MuX = MuX1;
    else
        MuX = zeros(3*PatchSize^2,1);
    end
elseif strcmp(ImageType, 'gray')==1
    MuX1 = sum(sum(IMin/255))/nnz(SampleMatrix)*ones(PatchSize^2,1);
    if IsSubMean
        MuX = sum(sum(IMin/255))/nnz(SampleMatrix);
    else
        MuX = 0;
    end
else
    MuX1 = zeros(size(IMin,3)*PatchSize^2,1);
    for layer = 1:size(IMin,3)
        MuX1((layer-1)*PatchSize^2+(1:PatchSize^2)) = sum(sum(IMin(:,:,layer)/255))/nnz(SampleMatrix(:,:,layer));
    end
    if IsSubMean
        MuX = MuX1;
    else
        MuX = zeros(size(IMin,3)*PatchSize^2,1);
    end
end

% Set Hyperparameters
c0=1e-6;
d0=1e-6;
e0=1e-6;
f0=1e-6;
sizeIMin = size(IMin);


for colj=1:PatchSize
    for rowi=1:PatchSize
        idexold = idex;
        idexNew = idexUpdate(sizeIMin(1:2),PatchSize,colj,rowi);
        idex = [idexold;idexNew];
        clear X_k
        Yflag = YflagUpdate(SampleMatrix,PatchSize,idex,ImageType,SparseCalculationT);
        X_k = XUpdate_MissingData(IMin,PatchSize,idex,MuX,Yflag,ImageType);
        
        [P,N] = size(X_k);
        
        %Sparsity Priors
        if strcmp(InitOption,'SVD')==1
            a0 = 1;
            b0 = N/8;
        else
            a0=1;
            b0=1;
        end
        
        %Initializations for new added patches
        if rowi==1 && colj==1
            [D,S,Z,phi,alpha,Pi] = InitMatrix(X_k,P,N,K,InitOption,UpdateOption,Yflag,IsSubMean,MuX1);
        else
            [S,Z] = SZUpdate(S,Z,rowi,idexNew,idexold,sizeIMin,PatchSize);
        end
        maxIt = IterPerRound(colj,rowi);
        X_k = Yflag.*(X_k - D*(Z.*S)');
        
        for iter=1:maxIt
            tic
            [X_k, D, Z, S] = SampleDZS_MissingData(X_k, D, Z, S, Pi, alpha, phi, P, K, N, Yflag, true, true, true, UpdateOption);
            Pi = SamplePi(Z,N,a0,b0);
            alpha   =   Samplealpha(S,e0,f0,Z,alpha);
            phi     =   Samplephi(X_k,c0,d0,Yflag);
            ittime=toc;
            
            NoiseVar(end+1) = sqrt(1/phi)*255;
            if ReduceDictSize && colj>2
                sumZ = sum(Z,1)';
                if min(sumZ)==0
                    Pidex = sumZ==0;
                    D(:,Pidex)=[];
                    K=size(D,2);
                    S(:,Pidex)=[];
                    Z(:,Pidex)=[];
                    Pi(Pidex)=[];
                end
            end
            
            if DispPSNR==1 || (rowi==8&&colj==8&&iter==maxIt)
                Iout    =   DenoiseOutput(D*(S.*Z)',sizeIMin,PatchSize,idex,MuX,ImageType);
                PSNR(end+1) = 20*log10(255/sqrt(mean((Iout(:)-IMin0(:)).^2)));
                disp(['round:',num2str([colj,rowi]),'    iter:', num2str(iter), '    time: ', num2str(ittime), '    ave_Z: ', num2str(mean(sum(Z,2))),'    M:', num2str(sum(Pi>1/1000)),'    PSNR:',num2str(PSNR(end)),'   NoiseVar:',num2str(NoiseVar(end)) ])
                save([IMname,'_',ImageType,'_Inpainting_',InitOption,'_',UpdateOption,'_',num2str(IsSubMean),'_',num2str(colj),'_',num2str(rowi)], 'Iout', 'SampleMatrix','D','PSNR','Pi','IMin','IMin0','phi','alpha','b0','NoiseVar');
            else
                disp(['round:',num2str([colj,rowi]),'    iter:', num2str(iter), '    time: ', num2str(ittime), '    ave_Z: ', num2str(mean(sum(Z,2))),'    M:', num2str(sum(Pi>1/1000)),'   NoiseVar:',num2str(NoiseVar(end)) ])
            end
            %whos S Z X_k Yflag
        end
    end
end
save([IMname,'_',ImageType,'_Inpainting_',InitOption,'_',UpdateOption,'_',num2str(IsSubMean)],'D','S','Z','SampleMatrix','phi','alpha','PSNR','IMin','IMin0','idex','Iout','Pi','b0','NoiseVar');
end