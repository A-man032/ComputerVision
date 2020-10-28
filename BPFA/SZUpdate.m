function [S,Z] = SZUpdate(S,Z,rowi,idexNew,idexold,sizeIMin,PatchSize)
%Initializing new S and Z with neighboring pacthes which have already been
%used for training in previous training rounds
%Version 1: 09/12/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
if rowi~=1
    TempDex = idexNew-1;
else
    TempDex = idexNew-(sizeIMin(1)-PatchSize+1);
end
Pos = zeros(length(TempDex),1);
for i=1:length(TempDex)
    Pos(i)=find(TempDex(i)==idexold,1,'last');
end
S=[S;S(Pos,:)];
Z=[Z;Z(Pos,:)];
end