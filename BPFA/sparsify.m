function X = sparsify(X)
%Squeeze out zero components in the sparse matrix
%Version 1: 10/26/2009
%Written by Mingyuan Zhou, Duke ECE, mz1@ee.duke.edu
[i,j,x] = find(X); %查找非零元素的索引和值
[m,n] = size(X);
X = sparse(i,j,x,m,n); %创建稀疏矩阵，将 X 的大小指定为 m×n。
end