% 04_train_models.m
% Train KNN (k sweep), SVM (RBF), FFNN (patternnet, trainscg) and save models

load('config.mat','params');
load('dataset.mat','Xtr','ytr');

% KNN sweep
bestK = params.knn_k_candidates(1);
bestAcc = 0;
for k = params.knn_k_candidates
    mdl = fitcknn(Xtr,ytr,'NumNeighbors',k);
    % evaluate via resubstitution? we'll store but final test acc computed later
    % compute 5-fold CV accuracy to choose k
    cvs = crossval(mdl,'KFold',5);
    loss = kfoldLoss(cvs);
    acc = 1 - loss;
    fprintf("KNN k=%d CV acc=%.3f\n",k,acc);
    if acc > bestAcc
        bestAcc = acc; bestK = k;
    end
end
knnModel = fitcknn(Xtr,ytr,'NumNeighbors',bestK);
fprintf("Selected KNN k=%d\n",bestK);

% SVM (ECOC with RBF)
t = templateSVM('KernelFunction','rbf','KernelScale','auto');
svmModel = fitcecoc(Xtr,ytr,'Learners',t,'Coding','onevsall');

% FFNN training
numUsers = numel(unique(ytr));
Ttr = full(ind2vec(ytr(:)', numUsers));

net = patternnet(params.ffnn_layers,'trainscg');
net.trainParam.epochs = params.ffnn_epochs;
net.trainParam.max_fail = params.ffnn_maxfail;
net.performFcn = 'crossentropy';
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:size(Xtr,1);
net.divideParam.valInd = [];
net.divideParam.testInd = [];
[net, tr] = train(net, Xtr', Ttr);

% save
if params.saveMat
    save('models.mat','knnModel','svmModel','net','bestK','params','-v7.3');
end
fprintf("Trained models saved.\n");
