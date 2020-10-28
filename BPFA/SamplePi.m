function Pi = SamplePi(Z,N,a0,b0)
%Sample Pi
%Version 1: 09/12/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
sumZ = full(sum(Z,1)'); %full()将稀疏矩阵转换为满存储
Pi = betarnd(sumZ+a0, b0+N-sumZ);
end