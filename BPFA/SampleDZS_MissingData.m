function [X_k, D, Z, S] = SampleDZS_MissingData(X_k, D, Z, S, Pi, alpha, phi, P, K, N, Yflag, Dsample, Zsample, Ssample,UpdateOption)
%Sample the Dictionary D, the Sparsity Pattern Z, and the Pesudo Weights S
%when there are missing data
%Version 1: 10/26/2009
%Version 2: 10/28/2009
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
%  Yflag: 
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
%Date: 10/24/2020
%------------------------------------------------------------------

if nargin<12
    Dsample = true;
end
if nargin<13
    Zsample = true;
end
if nargin<14
    Ssample = true;
end
if nargin<15
    UpdateOption = 'DkZkSk'; %UpdateOption is factor or global
end

if issparse(Yflag)
    sparseCalculation = true;
else
    sparseCalculation = false;
end


if strcmp(UpdateOption, 'DkZkSk')==1
    for k=1:K
        nnzk = nnz(Z(:,k));
        if nnzk>0
            if sparseCalculation
                X_k(:,Z(:,k)) = X_k(:,Z(:,k)) + sparse(spdiags(D(:,k),0,speye(P))*Yflag(:,Z(:,k)))*spdiags(S(Z(:,k),k),0,speye(nnzk));
            else
                X_k(:,Z(:,k)) = X_k(:,Z(:,k))+ Yflag(:,Z(:,k)).*(D(:,k)*S(Z(:,k),k)');
            end
        end
        
        if Dsample
            %Sample D
            sig_Dk = 1./(phi*Yflag(:,Z(:,k))*(S(Z(:,k),k).^2)+P);
            mu_D = phi*sig_Dk.*(X_k(:,Z(:,k))*S(Z(:,k),k));
            D(:,k) = mu_D + randn(P,1).*sqrt(sig_Dk);
            %clear sig_Dk mu_D
        end
        
        DTD = (D(:,k).^2)'*Yflag;
        
        if Zsample
            %Sample Z
            Sk = full(S(:,k));
            Sk(~Z(:,k)) = randn(N-nnz(Z(:,k)),1)*sqrt(1/alpha);
            temp =  - 0.5*phi*...
                ( (Sk.^2 ).*(DTD)' - 2*Sk.*(X_k'*D(:,k)) );
            temp = exp(temp)*Pi(k);
            Z(:,k) = sparse( rand(N,1) > ((1-Pi(k))./(temp+1-Pi(k))) );
            %clear Sk temp
        end
        
        nnzk = nnz(Z(:,k));
        
        if Ssample
            %Sample S
            sigS1 = 1./(alpha + phi*DTD(Z(:,k))');
            %S(Z(:,k),k) = randn(nnzk,1).*sqrt(sigS1)+ sigS1.*(phi*X_k(:,Z(:,k))'*D(:,k));
            %S(~Z(:,k),k) = 0;
            %S(~Z(:,k),k) = randn(N-nnzk,1)*sqrt(1/alpha);
            S(:,k) = sparse(find(Z(:,k)),1,randn(nnzk,1).*sqrt(sigS1)+ sigS1.*(phi*X_k(:,Z(:,k))'*D(:,k)), N,1);
            %clear sigS1
        end
        % X_k(:,Z(:,k)) = X_k(:,Z(:,k))- Yflag(:,Z(:,k)).*(D(:,k)*S(Z(:,k),k)');
        
        if nnzk>0
            if sparseCalculation
                X_k(:,Z(:,k)) = X_k(:,Z(:,k)) - sparse(spdiags(D(:,k),0,speye(P))*Yflag(:,Z(:,k)))*spdiags(S(Z(:,k),k),0,speye(nnzk));
            else
                X_k(:,Z(:,k)) = X_k(:,Z(:,k))- Yflag(:,Z(:,k)).*(D(:,k)*S(Z(:,k),k)');
            end
        end
    end
    
elseif strcmp(UpdateOption, 'DZS')==1
    %Sample D
    if Dsample
        for k=1:K
            nnzk = nnz(Z(:,k));
            if nnzk>0
                if sparseCalculation
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k)) + sparse(spdiags(D(:,k),0,speye(P))*Yflag(:,Z(:,k)))*spdiags(S(Z(:,k),k),0,speye(nnzk));
                else
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k))+ Yflag(:,Z(:,k)).*(D(:,k)*S(Z(:,k),k)');
                end
            end
            sig_Dk = 1./(phi*Yflag(:,Z(:,k))*(S(Z(:,k),k).^2)+P);
            mu_D = phi*sig_Dk.*(X_k(:,Z(:,k))*S(Z(:,k),k));
            D(:,k) = mu_D + randn(P,1).*sqrt(sig_Dk);
            if nnzk>0
                if sparseCalculation
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k)) - sparse(spdiags(D(:,k),0,speye(P))*Yflag(:,Z(:,k)))*spdiags(S(Z(:,k),k),0,speye(nnzk));
                else
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k))- Yflag(:,Z(:,k)).*(D(:,k)*S(Z(:,k),k)');
                end
            end
        end
        %clear sig_Dk mu_D
    end
    
    %Sample Z
    if Zsample
        for k=1:K
            nnzk = nnz(Z(:,k));
            if nnzk>0
                if sparseCalculation
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k)) + sparse(spdiags(D(:,k),0,speye(P))*Yflag(:,Z(:,k)))*spdiags(S(Z(:,k),k),0,speye(nnzk));
                else
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k))+ Yflag(:,Z(:,k)).*(D(:,k)*S(Z(:,k),k)');
                end
            end
            DTD = (D(:,k).^2)'*Yflag;
            Sk = full(S(:,k));
            Sk(~Z(:,k)) = randn(N-nnz(Z(:,k)),1)*sqrt(1/alpha);
            temp =  - 0.5*phi*...
                ( (Sk.^2 ).*(DTD)' - 2*Sk.*(X_k'*D(:,k)) );
            temp = exp(temp)*Pi(k);
            Z(:,k) = sparse( rand(N,1) > ((1-Pi(k))./(temp+1-Pi(k))) );
            nnzk = nnz(Z(:,k));
            if nnzk>0
                if sparseCalculation
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k)) - sparse(spdiags(D(:,k),0,speye(P))*Yflag(:,Z(:,k)))*spdiags(S(Z(:,k),k),0,speye(nnzk));
                else
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k))- Yflag(:,Z(:,k)).*(D(:,k)*S(Z(:,k),k)');
                end
            end
        end
        %clear Sk temp
    end
    Z = sparsify(Z);
    
    %Sample S
    if Ssample
        for k=1:K
            nnzk = nnz(Z(:,k));
            if nnzk>0
                if sparseCalculation
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k)) + sparse(spdiags(D(:,k),0,speye(P))*Yflag(:,Z(:,k)))*spdiags(S(Z(:,k),k),0,speye(nnzk));
                else
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k))+ Yflag(:,Z(:,k)).*(D(:,k)*S(Z(:,k),k)');
                end
            end
            DTD1 = (D(:,k).^2)'*Yflag(:,Z(:,k));
            sigS1 = 1./(alpha + phi*DTD1');
            %S(Z(:,k),k) = randn(nnzk,1).*sqrt(sigS1)+ sigS1.*(phi*X_k(:,Z(:,k))'*D(:,k));
            %S(~Z(:,k),k) = 0;
            %S(~Z(:,k),k) = randn(N-nnzk,1)*sqrt(1/alpha);
            S(:,k) = sparse(find(Z(:,k)),1,randn(nnzk,1).*sqrt(sigS1)+ sigS1.*(phi*X_k(:,Z(:,k))'*D(:,k)), N,1);
            if nnzk>0
                if sparseCalculation
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k)) - sparse(spdiags(D(:,k),0,speye(P))*Yflag(:,Z(:,k)))*spdiags(S(Z(:,k),k),0,speye(nnzk));
                else
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k))- Yflag(:,Z(:,k)).*(D(:,k)*S(Z(:,k),k)');
                end
            end
        end
        %clear sigS1
    end
    S = sparsify(S);    
elseif strcmp(UpdateOption, 'DZkSk')==1
    %Sample D
    if Dsample
        for k=1:K
            nnzk = nnz(Z(:,k));
            if nnzk>0
                if sparseCalculation
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k)) + sparse(spdiags(D(:,k),0,speye(P))*Yflag(:,Z(:,k)))*spdiags(S(Z(:,k),k),0,speye(nnzk));
                else
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k))+ Yflag(:,Z(:,k)).*(D(:,k)*S(Z(:,k),k)');
                end
            end
            sig_Dk = 1./(phi*Yflag(:,Z(:,k))*(S(Z(:,k),k).^2)+P);
            mu_D = phi*sig_Dk.*(X_k(:,Z(:,k))*S(Z(:,k),k));
            D(:,k) = mu_D + randn(P,1).*sqrt(sig_Dk);
            if nnzk>0
                if sparseCalculation
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k)) - sparse(spdiags(D(:,k),0,speye(P))*Yflag(:,Z(:,k)))*spdiags(S(Z(:,k),k),0,speye(nnzk));
                else
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k))- Yflag(:,Z(:,k)).*(D(:,k)*S(Z(:,k),k)');
                end
            end
        end
        %clear sig_Dk mu_D
    end
    
    %Sample Z and S
    if Zsample && Ssample
        for k=1:K
            nnzk = nnz(Z(:,k));
            if nnzk>0
                if sparseCalculation
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k)) + sparse(spdiags(D(:,k),0,speye(P))*Yflag(:,Z(:,k)))*spdiags(S(Z(:,k),k),0,speye(nnzk));
                else
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k))+ Yflag(:,Z(:,k)).*(D(:,k)*S(Z(:,k),k)');
                end
            end
            DTD = (D(:,k).^2)'*Yflag;
            Sk = full(S(:,k));
            Sk(~Z(:,k)) = randn(N-nnz(Z(:,k)),1)*sqrt(1/alpha);
            temp =  - 0.5*phi*...
                ( (Sk.^2 ).*(DTD)' - 2*Sk.*(X_k'*D(:,k)) );
            temp = exp(temp)*Pi(k);
            Z(:,k) = sparse( rand(N,1) > ((1-Pi(k))./(temp+1-Pi(k))) );
            nnzk = nnz(Z(:,k));                        
            sigS1 = 1./(alpha + phi*DTD(Z(:,k))');
            %S(Z(:,k),k) = randn(nnzk,1).*sqrt(sigS1)+ sigS1.*(phi*X_k(:,Z(:,k))'*D(:,k));
            %S(~Z(:,k),k) = 0;
            %S(~Z(:,k),k) = randn(N-nnzk,1)*sqrt(1/alpha);
            S(:,k) = sparse(find(Z(:,k)),1,randn(nnzk,1).*sqrt(sigS1)+ sigS1.*(phi*X_k(:,Z(:,k))'*D(:,k)), N,1);
            if nnzk>0
                if sparseCalculation
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k)) - sparse(spdiags(D(:,k),0,speye(P))*Yflag(:,Z(:,k)))*spdiags(S(Z(:,k),k),0,speye(nnzk));
                else
                    X_k(:,Z(:,k)) = X_k(:,Z(:,k))- Yflag(:,Z(:,k)).*(D(:,k)*S(Z(:,k),k)');
                end
            end
        end
        %clear sigS1
    end
    Z = sparsify(Z);
    S = sparsify(S);
end




end