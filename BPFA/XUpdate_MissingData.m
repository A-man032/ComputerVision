
function X = XUpdate_MissingData(IMin,PatchSize,idex,MuX,Yflag,ImageType)
%Update the training data taken into consideration
%Version 1: 09/12/2009
%Version 2: 10/26/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
%------------------------------------------------------------------
%Input:
%  IMin: noise image (3-dimensional)
%  PatchSize: patch size, 8 is commonly used.
%  idex: the index of the patches in the training data
%  MuX: the globle mean 全局均值
%  Yflag: 
%  ImageType: the type of image
%Output:
%  X: the training data taken into consideration (2-dimensional)
%Written by Man Yang, ym@bupt.edu.cn
%Date: 10/19/2020
%------------------------------------------------------------------

if strcmp(ImageType,'rgb')==1
    X =  [im2col(IMin(:,:,1)/255,[PatchSize,PatchSize],'sliding'); %将图像块重新排列成列
        im2col(IMin(:,:,2)/255,[PatchSize,PatchSize],'sliding');
        im2col(IMin(:,:,3)/255,[PatchSize,PatchSize],'sliding')]; % X: 二维矩阵
    %im2col(A,[m n],block_type) 当block_type为sliding时，以子块滑动的方式将A分解成m×n的子矩阵，并将分解以后的子矩阵沿列的方向转换成B的列。
    %当block_type为distinct时，将A沿列的方向分解为互不重叠的子矩阵，并将分解以后的子矩阵沿列的方向转换成B的列，若不足m×n，以0补足
    X = X(:,idex).*Yflag; % .*: 对应元素相乘
    X = X - repmat(MuX,1,length(idex)).*Yflag; %substract the globle mean
else
    X = im2col(IMin/255,[PatchSize,PatchSize],'sliding');
    %X = X(:,idex)-repmat(MuX,1,length(idex));
    X = X(:,idex).*Yflag-MuX.*Yflag;    
end