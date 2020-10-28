function [X_k, D, Z, S] = SampleDZS(X_k, D, Z, S, Pi, alpha, phi, P, K, N, Dsample, Zsample, Ssample,UpdateOption)
%Sample the Dictionary D, the Sparsity Pattern Z, and the Pesudo Weights S
%when there are no missing data.
%Version 1: 10/21/2009
%Version 2: 10/26/2009
%Version 3: 10/28/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
%-------------------- ----------------------------------------------
%Input:
%  X_k: noise matrix
%  D: dictionary learned from the noisy image(2-dimensional)
%  Z: binary indicators for basis usage, follow Bernoulli Distribution
%  S: basis coefficients
%  Pi: the probabilities for each dictionary entries to be used
%  alpha: precision for S
%  phi: noise precision
%  P: the size of X'rows
%  K: predefined dictionary size
%  N: the size of X'column
%  Dsample: (false)sampling D if TRUE
%  Zsample: (true)sampling Z if TRUE
%  Ssample: (true)sampling S if TRUE
%  UpdateOption: In the order of 'DkZkSk','DZS', or 'DZkSk'. the
%                computation requirments are 'DkZkSk'<'DZkSk'<'DZS'.
%Output:
%  X_k: noise matrix after sampling
%  D: 
%  Z: Z's non-zero sparse matrix
%  S: S's non-zero sparse matrix
%Written by Man Yang, ym@bupt.edu.cn
%Date: 10/21/2020
%------------------------------------------------------------------

if nargin<11
    Dsample = true;
end
if nargin<12
    Zsample = true;
end
if nargin<13
    Ssample = true;
end
if nargin<15
    UpdateOption = 'DkZkSk'; %UpdateOption is factor or global
end
if strcmp(UpdateOption, 'DkZkSk')==1
    for k=1:K
        X_k(:,Z(:,k)) = X_k(:,Z(:,k))+ D(:,k)*S(Z(:,k),k)'; %X=D*omega+epsilon(与k有关)
        if Dsample
            %Sample D
            sig_Dk = 1./(phi*sum(S(Z(:,k),k).^2)+P); %./对应元素相除 .^2矩阵每个元素求平方
            mu_D = phi*sig_Dk* (X_k(:,Z(:,k))*S(Z(:,k),k)); %向量
            D(:,k) = mu_D + randn(P,1)*sqrt(sig_Dk); %取样
            %clear sig_Dk mu_D
        end
        
        DTD = sum(D(:,k).^2); %dk转置*dk
        
        if Zsample
            %Sample Z
            Sk = full(S(:,k)); %将稀疏矩阵转换为满存储 更改矩阵的存储格式，并比较存储要求。
                               %创建一个随机稀疏矩阵。在 MATLAB® 中，稀疏矩阵的显示会忽略所有零，只显示非零元素的位置和值。
            Sk(~Z(:,k)) = randn(N-nnz(Z(:,k)),1)*sqrt(1/alpha); %nnz(X)返回矩阵X中的非零元素数
            temp =  - 0.5*phi*...
                ( (Sk.^2 )*DTD - 2*Sk.*(X_k'*D(:,k)) );
            temp = exp(temp)*Pi(k); %p1
            Z(:,k) = sparse( rand(N,1) > ((1-Pi(k))./(temp+1-Pi(k))) ); %??
            %clear Sk temp
        end
        
        if Ssample
            %Sample S
            sigS1 = 1/(alpha + phi*DTD); %Zik=1
            %S(Z(:,k),k) = randn(nnz(Z(:,k)),1)*sqrt(sigS1)+ sigS1*(phi*(X_k(:,Z(:,k))'*D(:,k)));
            %S(~Z(:,k),k) = randn(N-nnz(Z(:,k)),1)*sqrt(1/alpha);
            S(:,k) = sparse(find(Z(:,k)),1,randn(nnz(Z(:,k)),1)*sqrt(sigS1)+ sigS1*(phi*(X_k(:,Z(:,k))'*D(:,k))),N,1);
            %稀疏矩阵的行下标是Z中非零元素的下标
            %clear sigS1
        end
        X_k(:,Z(:,k)) = X_k(:,Z(:,k))- D(:,k)*S(Z(:,k),k)';
    end
    Z = sparsify(Z);
    S = sparsify(S);
elseif strcmp(UpdateOption, 'DZS')==1
    if Dsample
        for k=1:K
            X_k(:,Z(:,k)) = X_k(:,Z(:,k))+ D(:,k)*S(Z(:,k),k)';
            sig_Dk = 1./(phi*sum(S(Z(:,k),k).^2)+P);
            mu_D = phi*sig_Dk* (X_k(:,Z(:,k))*S(Z(:,k),k));
            D(:,k) = mu_D + randn(P,1)*sqrt(sig_Dk);
            X_k(:,Z(:,k)) = X_k(:,Z(:,k))- D(:,k)*S(Z(:,k),k)';
        end
        %clear sig_Dk mu_D
    end
    
    %Sample Z
    if Zsample
        for k=1:K
            X_k(:,Z(:,k)) = X_k(:,Z(:,k))+ D(:,k)*S(Z(:,k),k)';
            DTD = sum(D(:,k).^2);
            Sk = full(S(:,k));
            Sk(~Z(:,k)) = randn(N-nnz(Z(:,k)),1)*sqrt(1/alpha);
            temp =  - 0.5*phi*...
                ( (Sk.^2 )*DTD - 2*Sk.*(X_k'*D(:,k)) );
            temp = exp(temp)*Pi(k);
            Z(:,k) = sparse( rand(N,1) > ((1-Pi(k))./(temp+1-Pi(k))) );
            X_k(:,Z(:,k)) = X_k(:,Z(:,k))- D(:,k)*S(Z(:,k),k)';
        end
        %clear Sk temp
    end
    Z = sparsify(Z);
    
    %Sample S
    if Ssample
        for k=1:K
            X_k(:,Z(:,k)) = X_k(:,Z(:,k))+ D(:,k)*S(Z(:,k),k)';
            DTD = sum(D(:,k).^2);
            sigS1 = 1/(alpha + phi*DTD);
            %S(Z(:,k),k) = randn(nnz(Z(:,k)),1)*sqrt(sigS1)+ sigS1*(phi*(X_k(:,Z(:,k))'*D(:,k)));
            %S(~Z(:,k),k) = randn(N-nnz(Z(:,k)),1)*sqrt(1/alpha);
            S(:,k) = sparse(find(Z(:,k)),1,randn(nnz(Z(:,k)),1)*sqrt(sigS1)+ sigS1*(phi*(X_k(:,Z(:,k))'*D(:,k))),N,1);
            X_k(:,Z(:,k)) = X_k(:,Z(:,k))- D(:,k)*S(Z(:,k),k)';
        end
        %clear sigS1
    end
    S = sparsify(S);
elseif strcmp(UpdateOption, 'DZkSk')==1
    if Dsample
        for k=1:K
            X_k(:,Z(:,k)) = X_k(:,Z(:,k))+ D(:,k)*S(Z(:,k),k)';
            sig_Dk = 1./(phi*sum(S(Z(:,k),k).^2)+P);
            mu_D = phi*sig_Dk* (X_k(:,Z(:,k))*S(Z(:,k),k));
            D(:,k) = mu_D + randn(P,1)*sqrt(sig_Dk);
            X_k(:,Z(:,k)) = X_k(:,Z(:,k))- D(:,k)*S(Z(:,k),k)';
        end
        %clear sig_Dk mu_D
    end
    
    %Sample Z and S
    if Zsample && Ssample
        for k=1:K
            X_k(:,Z(:,k)) = X_k(:,Z(:,k))+ D(:,k)*S(Z(:,k),k)';
            DTD = sum(D(:,k).^2);
            Sk = full(S(:,k));
            Sk(~Z(:,k)) = randn(N-nnz(Z(:,k)),1)*sqrt(1/alpha);
            temp =  - 0.5*phi*...
                ( (Sk.^2 )*DTD - 2*Sk.*(X_k'*D(:,k)) );
            temp = exp(temp)*Pi(k);
            Z(:,k) = sparse( rand(N,1) > ((1-Pi(k))./(temp+1-Pi(k))) );            
            sigS1 = 1/(alpha + phi*DTD);
            %S(Z(:,k),k) = randn(nnz(Z(:,k)),1)*sqrt(sigS1)+ sigS1*(phi*(X_k(:,Z(:,k))'*D(:,k)));
            %S(~Z(:,k),k) = randn(N-nnz(Z(:,k)),1)*sqrt(1/alpha);
            S(:,k) = sparse(find(Z(:,k)),1,randn(nnz(Z(:,k)),1)*sqrt(sigS1)+ sigS1*(phi*(X_k(:,Z(:,k))'*D(:,k))),N,1);
            X_k(:,Z(:,k)) = X_k(:,Z(:,k))- D(:,k)*S(Z(:,k),k)';
        end
        %clear sigS1
    end
    Z = sparsify(Z);
    S = sparsify(S);
end
end