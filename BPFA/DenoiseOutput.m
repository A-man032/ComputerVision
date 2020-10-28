function  Iout = DenoiseOutput(Xo,sizeIMin,PatchSize,idex,MuX,ImageType)
%reconstruct the image
%Version 1: 09/12/2009
%Version 2: 10/21/2009
%Version 3: 10/26/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu

if nargin<6
    ImageType = 'gray';
end
if strcmp(ImageType,'gray')==1
    Xo = Xo+MuX;
else
    PatchSize2 = PatchSize^2;
    Xo = Xo+repmat(MuX,1,size(Xo,2));
end
Io = zeros(sizeIMin);
Weight = zeros(sizeIMin);
[cols,rows] = ind2sub(sizeIMin-PatchSize+1,idex);
for i=1:length(idex);
    Pos1  = cols(i)+(0:PatchSize-1);
    Pos2 = rows(i)+(0:PatchSize-1);
    if strcmp(ImageType,'gray')==1
        Io(Pos1,Pos2) = Io(Pos1,Pos2) + reshape(Xo(:,i),PatchSize,PatchSize);      
        Weight(Pos1,Pos2)= Weight(Pos1,Pos2) + 1;
    else
        for rgb=1:3
            Io(Pos1,Pos2,rgb) = Io(Pos1,Pos2,rgb) + reshape(Xo((rgb-1)*PatchSize2+(1:PatchSize2),i),PatchSize,PatchSize);
            Weight(Pos1,Pos2,rgb)= Weight(Pos1,Pos2,rgb) + 1;
        end
    end
end
Iout = 255*Io./Weight;
Iout = (max(min(Iout,255),0));
end