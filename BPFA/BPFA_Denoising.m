
function [ave,Iout,D,S,Z,idex,Pi,NoiseVar,alpha,phi,PSNR,PSNRave] = BPFA_Denoising(IMin, PatchSize, K, InitOption,UpdateOption,IsSubMean, IterPerRound, ReduceDictSize, IMin0,sigma,IMname,DispPSNR)
%------------------------------------------------------------------
% The BPFA grayscale image denoising program for the paper:
% "Non-Parametric Bayesian Dictionary Learning for Sparse Image
% Representations," Neural Information Processing Systems (NIPS), 2009.
% Coded by: Mingyuan Zhou, ECE, Duke University, mz1@ee.duke.edu
% Version 0: 06/14/2009
% Version 1: 09/13/2009
% Version 2: 10/21/2009
% Version 3: 10/28/2009
%------------------------------------------------------------------
% Input:
%   IMin: noisy image. 3-dimensional
%   PatchSize: patch size, 8 is commonly used.
%   K: predefined dictionary size.
%   InitOption: SVD,SVD1,RAND,Kmeans
%   UpdateOption: In the order of 'DkZkSk','DZS', or 'DZkSk'. the
%   computation requirments are 'DkZkSk'<'DZkSk'<'DZS'.
%   IsSubMean: substract the globle mean if it is TRUE
%   IterPerRound: maximum iteration in each round.
%   ReduceDictSize: reduce the dictionary size during learning if it is
%   TRUE
%   IMin0: original noise-free image (only used for PSNR calculation and
%   would not affect the denoising results).
%   sigma: noise variance (has no effect on the denoising results).
%   IMname: image name (has no effect on the deoising results).
%   DispPSNR: calculate and display the instant PSNR during learning if it
%   is TRUE
% Output:
%   ave: denoised image (averaged output)
%   Iout: denoised image (sample output)
%   D: dictionary learned from the noisy image
%   S: basis coefficients
%   Z: binary indicators for basis usage
%   idex: patch index
%   Pi: the probabilities for each dictionary entries to be used
%   NoiseVar: estimated noise variance
%   alpha: precision for S
%   phi: noise precision
%   PSNR: peak signal-to-noise ratio
%   PSNRave: PSNR of ave.Iout
%------------------------------------------------------------------

%补齐参数
if nargin<2
    PatchSize=8;
end
if nargin < 3
    K=256;
end
if nargin <4
    InitOption='SVD';
end
if nargin <5
    UpdateOption='factor';
end
if nargin <6
    IsSubMean = false;
end
if nargin <7
    IterPerRound = ones(PatchSize,PatchSize);
    IterPerRound(1,1) = 10;
    IterPerRound(1,2:end) = 2;
    IterPerRound(end,end) = 30;
end
if nargin<8
    ReduceDictSize = false;
end
if nargin < 9
    IMin0=IMin; %origial image=noise image
end
if nargin < 10
    sigma=[];
end
if nargin <11
    IMname=[];
end
if nargin <12
    DispPSNR = true;
end


sizeIMin = size(IMin);
idex=[];
PSNR=[];
NoiseVar=[];

if IsSubMean %IsSubMean: substract the globle mean if it is TRUE
    MuX = mean(mean(IMin/255)); %全局均值 求了两次均值，MuX是个数
else
    MuX = 0;
end

% Set Hyperparameters
c0=1e-6;
d0=1e-6;
e0=1e-6;
f0=1e-6;

ave.Iout = zeros(size(IMin));
ave.Count = 0;

for colj=1:PatchSize
    for rowi=1:PatchSize
        idexold = idex; 
        idexNew = idexUpdate(sizeIMin,PatchSize,colj,rowi);
        idex = [idexold;idexNew]; %按行合并两个向量12*1(1-dimension)
        clear X_k
        X_k = XUpdate(IMin,PatchSize,idex,MuX); %2-dimension
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
            [D,S,Z,phi,alpha,Pi] = InitMatrix(X_k,P,N,K,InitOption,UpdateOption);
        else
            [S,Z] = SZUpdate(S,Z,rowi,idexNew,idexold,sizeIMin,PatchSize);
        end
        maxIt = IterPerRound(colj,rowi);
        X_k = X_k - D*(Z.*S)';
        for iter=1:maxIt
            tic
            [X_k, D, Z, S] = SampleDZS(X_k, D, Z, S, Pi, alpha, phi, P, K, N, true, true, true, UpdateOption);
            Pi = SamplePi(Z,N,a0,b0);
            alpha = Samplealpha(S,e0,f0,Z,alpha);
            phi = Samplephi(X_k,c0,d0);
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
           
            if DispPSNR==1 || (rowi==8&&colj==8)
                Iout    =   DenoiseOutput(D*(S.*Z)',sizeIMin,PatchSize,idex,MuX);
                ave.Count = ave.Count + 1;
                PSNRave=0;
                if ave.Count==1
                    ave.Iout = Iout;
                else
                    ave.Iout= 0.85*ave.Iout+0.15*Iout;
                    PSNRave = 20*log10(255/sqrt(mean((ave.Iout(:)-IMin0(:)).^2)));
                end               
                PSNR(end+1) = 20*log10(255/sqrt(mean((Iout(:)-IMin0(:)).^2)));
                disp(['round:',num2str([colj,rowi]),'    iter:', num2str(iter), '    time: ', num2str(ittime), '    ave_Z: ', num2str(mean(sum(Z,2))),'    M:', num2str(sum(Pi>1/1000)),'    PSNR:',num2str(PSNR(end)),'    PSNRave:',num2str(PSNRave),'   NoiseVar:',num2str(NoiseVar(end)) ])
                save([IMname,'_Denoising_',InitOption,'_',UpdateOption,'_',num2str(IsSubMean),'_',num2str(sigma),'_',num2str(colj),'_',num2str(rowi)], 'Iout', 'D','PSNR','Pi','IMin','IMin0','phi','alpha','NoiseVar','idex','ave');
            else
                disp(['round:',num2str([colj,rowi]),'    iter:', num2str(iter), '    time: ', num2str(ittime), '    ave_Z: ', num2str(mean(sum(Z,2))),'    M:', num2str(sum(Pi>1/1000)),'   NoiseVar:',num2str(NoiseVar(end)) ])
            end            
        end      
    end
end
save( [IMname,'_Denoising_',InitOption,'_',UpdateOption,'_',num2str(IsSubMean),num2str(sigma)], 'Iout', 'D','S','Z','PSNR','Pi','IMin','IMin0','phi','alpha','NoiseVar','idex','ave');
end