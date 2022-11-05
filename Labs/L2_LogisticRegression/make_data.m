function [X, Y] = make_data(p0, p1, varargin)

% var = 0.5;  % Variance
% N = 300;    % How many points in each class
Defaults = {1, 300}; % Defaults for var and N
Defaults(1:nargin-2) = varargin;
var = Defaults{1};
N   = Defaults{2};

points0 = p0 + var*randn(N,2);
points1 = p1 + var*randn(N,2);
targets0 = zeros(N,1);
targets1 = ones(N,1);
Points = [points0; points1];
Targets = [targets0; targets1];
Data = [Points Targets];
Data = Data(randperm(size(Data, 1)), :); % shuffle rows
X = Data(:,1:2);
Y = Data(:,3);
gscatter(X(:,1),X(:,2),Y)
%save('LogisticReg.mat', 'X', 'Y')
end
