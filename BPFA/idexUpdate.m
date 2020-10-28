
function idexNew = idexUpdate(sizeIMin, PatchSize,colj,rowi)
%Update the index of the patches in the training data
%Version 1: 09/12/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
%------------------------------------------------------------------
%Input:
%  sizeIMin: the size of noise image.
%  PatchSize: patch size, 8 is commonly used.
%  colj: j th column
%  rowi: i th row
%OutPut:
%  idexNew: the index of the patches in the training data(a vector of the
%           index of non-zero elements in matrix idexMat)
%Written by Man Yang, ym@bupt.edu.cn
%Date: 10/19/2020
%------------------------------------------------------------------

%每次抽取相同数目的元素，抽取规则如下
idexMat=zeros(sizeIMin-PatchSize+1); %返回全零的(Ny-B+1)*(Nx-B+1)的矩阵
if colj==1 && rowi==1
    idexMat([rowi:PatchSize:end-1,end],[colj:PatchSize:end-1,end])=1;
elseif colj==1 && rowi~=1
    idexMat(rowi:PatchSize:end,[colj:PatchSize:end-1,end])=1;
elseif colj~=1 && rowi==1
    idexMat([rowi:PatchSize:end-1,end],colj:PatchSize:end)=1;
else
    idexMat(rowi:PatchSize:end,colj:PatchSize:end)=1;
end
idexNew = find(idexMat); %返回一个包含数组 idexMat 中每个非零元素的线性索引的向量
end