% 03_build_dataset.m
% Clean features, remove constant cols, normalize, split FD->train MD->test

load('config.mat','params');
load('features_raw.mat','X','allUsers','allSess');

% remove columns with near-zero variance
colstd = std(X,[],1);
keep = colstd > 1e-6;
X = X(:,keep);
fprintf("Dropped %d constant columns. New dim = %d\n", sum(~keep), size(X,2));

% replace NaN/Inf
X(~isfinite(X)) = 0;

% normalization (store params)
switch lower(params.normalize_method)
    case 'minmax'
        xmin = min(X,[],1); xmax = max(X,[],1);
        Xn = (X - xmin) ./ (xmax - xmin + eps);
        normParams.method = 'minmax'; normParams.xmin = xmin; normParams.xmax = xmax;
    case 'zscore'
        mu = mean(X,1); s = std(X,[],1);
        Xn = (X - mu) ./ (s + eps);
        normParams.method = 'zscore'; normParams.mu = mu; normParams.s = s;
    otherwise
        error("Unknown normalization");
end

% split
trainIdx = (allSess == 1);
testIdx = (allSess == 2);

Xtr = Xn(trainIdx,:);
ytr = allUsers(trainIdx)';
Xte = Xn(testIdx,:);
yte = allUsers(testIdx)';

fprintf("Train: %d, Test: %d\n", size(Xtr,1), size(Xte,1));
if params.saveMat
    save('dataset.mat','Xtr','ytr','Xte','yte');
    save('norm_params.mat','normParams');
end
