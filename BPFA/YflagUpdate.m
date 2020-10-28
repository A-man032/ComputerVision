function Yflag = YflagUpdate(SampleMatrix,PatchSize,idex,ImageType,SparseCalculationT)
%Update the mask matrix asccociated with X
%Version 1: 09/12/2009
%Version 2: 10/26/2009
%Version 3: 10/28/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
if nargin<4
    ImageType='gray';
end
if nargin<5
    SparseCalculationT = 0.8;
end
if strcmp(ImageType,'gray')==1
    Yflag = logical(im2col(SampleMatrix,[PatchSize,PatchSize],'sliding'));
else    
    Yflag=[logical(im2col(SampleMatrix(:,:,1),[PatchSize,PatchSize],'sliding'));
           logical(im2col(SampleMatrix(:,:,2),[PatchSize,PatchSize],'sliding'));
           logical(im2col(SampleMatrix(:,:,3),[PatchSize,PatchSize],'sliding'))];
end
Yflag = Yflag(:,idex);
if nnz(Yflag)/numel(Yflag) < SparseCalculationT
    Yflag = sparse(Yflag);
end
end