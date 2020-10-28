function [D,S,Z,phi,alpha,Pi] = InitMatrix(X,P,N,K,InitOption,UpdateOption, Yflag,IsSubMean,MuX1)
%Initialization
%Version 1: 09/12/2009
%Version 2: 10/21/2009
%Version 3: 10/26/2009
%Version 4: 10/28/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
%------------------------------------------------------------------
%Input:
%  X: the training data taken into consideration (2-dimensional: P*N)
%  P: the size of X'rows
%  N: the size of X'column
%  K: predefined dictionary size
%  InitOption: SVD,SVD1,RAND,Kmeans
%  UpdateOption: In the order of 'DkZkSk','DZS', or 'DZkSk'. the
%   computation requirments are 'DkZkSk'<'DZkSk'<'DZS'.
%  Yflag: 
%  IsSubMean: substract the globle mean if it is TRUE
%  MuX1: 
%Output:
%  D: dictionary matrix(2-dimensional)
%  S: 
%  Z:
%  phi: 
%  alpha:
%  Pi: 
%Written by Man Yang, ym@bupt.edu.cn
%Date: 10/20/2020
%------------------------------------------------------------------

if nargin<7
    Yflag=[];
    IsSubMean = [];
    MuX1 = [];
else
    if ~IsSubMean 
        X = X.*Yflag + repmat(MuX1,1,N).*(~Yflag);
    end
end
phi = 1/((25/255)^2);
alpha = 1;
if strcmp(InitOption,'SVD')==1 || strcmp(InitOption,'SVD1')==1
    [U_1,S_1,V_1] = svd(full(X),'econ'); %奇异值分解
    if P<=K %行数<=字典大小
        D = zeros(P,K);
        D(:,1:P) = U_1*S_1; %仅计算前p列
        S = zeros(N,K);
        S(:,1:P) = V_1;
    else %行数>字典大小
        D =  U_1*S_1;
        D = D(1:P,1:K);
        S = V_1;
        S = S(1:N,1:K);
    end
    if strcmp(InitOption,'SVD')==1
        Z = true(N,K);
        Pi = 0.5*ones(K,1);
    else
        Z = sparse(false(N,K));
        Pi = 0.01*ones(K,1);
    end
elseif strcmp(InitOption,'RAND')==1
    D=randn(P,K)/sqrt(P);
    S=randn(N,K);
    Z = sparse(false(N,K)); %创建稀疏矩阵
    Pi = 0.01*ones(K,1);
elseif  strcmp(InitOption,'Kmeans')==1
    S = ones(N,K);
    Z = sparse(false(N,K));
    if N<K
        Xtemp=zeros(P,K);
        Xtemp(:,1:N)=X;
    else
        Xtemp=X;
    end
    [idx,ctrs] = kmeans(Xtemp',K,'emptyaction','singleton'); %Xtemp'表示Xtemp的共轭转置
    D  = ctrs';
    Pi = 0.01*ones(K,1);
    if nargin<7
        [X_k, D, Z, S] = SampleDZS(X - D*(Z.*S)', D, Z, S, Pi, alpha, phi, K, false, true, true,UpdateOption);
    else
        [X_k, D, Z, S] = SampleDZS_MissingData( (X - D*(Z.*S)').*Yflag, D, Z, S, Pi, alpha, phi, P, K, N, Yflag, false, true, true,UpdateOption);
    end
end
end





