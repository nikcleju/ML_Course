N = 300;
var = 0.5;
points0 = var*randn(N,2);
points1 = 1 + var*randn(N,2);
targets0 = zeros(N,1);
targets1 = ones(N,1);
Points = [points0; points1];
Targets = [targets0; targets1];
Data = [Points Targets];
Data = Data(randperm(size(Data, 1)), :); % shuffle rows
X = Data(:,1:2);
Y = Data(:,3);
%scatter(X(:,1), X(:,2),2, ['r','g'])
%scatter(X(Y==0,1), X(Y==0,2), X(Y==1,1), X(Y==1,2))
gscatter(X(:,1),X(:,2),Y)
save('LogisticReg.mat', 'X', 'Y')