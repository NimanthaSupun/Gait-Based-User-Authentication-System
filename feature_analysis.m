% 07_feature_analysis.m
% PCA, ANOVA, intra/inter distances for feature uniqueness/variability

load('features_raw.mat','X','allUsers','allSess');
% If features were cleaned in build step, optionally load cleaned Xn
if exist('dataset.mat','file')
    load('dataset.mat','Xtr','ytr','Xte','yte');
end

% PCA on raw X
[coeff,score,~,~,explained] = pca(X);
fprintf("PCA: PC1 %.2f%% PC2 %.2f%% variance\n", explained(1), explained(2));
figure('Name','PCA 2D'); gscatter(score(:,1),score(:,2),allUsers);
title('PCA of feature vectors (raw)');

% ANOVA per feature
pvals = zeros(1,size(X,2));
for fi=1:size(X,2)
    pvals(fi) = anova1(X(:,fi), allUsers, 'off');
end
fprintf("ANOVA: %d/%d features significant (p<0.05)\n", sum(pvals<0.05), numel(pvals));

% intra/inter distances
intra = []; inter = [];
uList = unique(allUsers);
for u = uList
    Xu = X(allUsers==u, :);
    Xu_other = X(allUsers~=u,:);
    if size(Xu,1) > 1
        intra = [intra; pdist(Xu)]; %#ok<AGROW>
    end
    inter = [inter; pdist2(Xu, Xu_other)]; %#ok<AGROW>
end
fprintf("Mean intra-class dist: %.4f Mean inter-class dist: %.4f\n", mean(intra), mean(inter));

if params.saveMat
    save('feature_analysis.mat','coeff','score','explained','pvals','intra','inter','-v7.3');
end
