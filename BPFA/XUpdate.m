
function X = XUpdate(IMin,PatchSize,idex,MuX)
%Update the training data taken into consideration
%Version 1: 09/12/2009
%Version 2: 10/21/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
X = im2col(IMin/255,[PatchSize,PatchSize],'sliding');
%X = X(:,idex)-repmat(MuX,1,length(idex));
X = X(:,idex)-MuX;
end