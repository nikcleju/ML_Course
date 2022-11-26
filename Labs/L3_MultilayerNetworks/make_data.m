function [X, Y] = make_data(centers, varargin)

% var = 0.5;  % Variance
% N = 300;    % How many points in each class
Defaults = {1, 300}; % Defaults for var and N
Defaults(1:nargin-2) = varargin;
var = Defaults{1};
N   = Defaults{2};

points = [];
targets = [];
for irow=1:size(centers,1)
	points = [points;  centers(irow,:) + var*randn(N,2)];
	targets = [targets; irow*ones(N,1)];
end	
Data = [points targets];
Data = Data(randperm(size(Data, 1)), :); % shuffle rows
X = Data(:,1:2);
Y = Data(:,3);
gscatter(X(:,1),X(:,2),Y)
end
