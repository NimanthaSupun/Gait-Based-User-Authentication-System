% 05_evaluate.m
% Evaluate models on test set, compute metrics, save confusion matrix figure

clear; clc;
load('dataset.mat','Xte','yte');
load('models.mat','knnModel','svmModel','net','bestK','params');

%% ===========================
%%  Create results folder
%% ===========================
resultsFolder = fullfile(pwd, "results");
if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end


%% ===========================
%%   PREDICTIONS
%% ===========================
pred_knn = predict(knnModel, Xte);
pred_svm = predict(svmModel, Xte);
scores_nn = net(Xte')';      % (N × C)
[~, pred_nn] = max(scores_nn,[],2);


%% ===========================
%%   CLASSIFICATION ACCURACY
%% ===========================
acc_knn = mean(pred_knn == yte);
acc_svm = mean(pred_svm == yte);
acc_nn  = mean(pred_nn == yte);

fprintf("\n======== CLASSIFICATION ACCURACY ========\n");
fprintf("KNN  (k=%d) Accuracy: %.2f%%\n", bestK, acc_knn*100);
fprintf("SVM  (RBF) Accuracy: %.2f%%\n", acc_svm*100);
fprintf("FFNN Accuracy: %.2f%%\n", acc_nn*100);


%% ===========================
%%   CONFUSION MATRIX (FFNN)
%% ===========================
figure('Name','Confusion Matrix - FFNN');
cm = confusionchart(yte, pred_nn);
cm.Title = 'Confusion Matrix - FFNN';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% Save confusion matrix
cmFigPath = fullfile(resultsFolder, "confusion_ffnn.png");
saveas(gcf, cmFigPath);
fprintf("Saved FFNN confusion matrix → %s\n", cmFigPath);


%% ===========================
%% PRECISION / RECALL / F1
%% ===========================
conf = confusionmat(yte, pred_nn);
C = size(conf,1);
prec = zeros(C,1); rec = zeros(C,1); f1 = zeros(C,1);

for u=1:C
    TP = conf(u,u);
    FP = sum(conf(:,u)) - TP;
    FN = sum(conf(u,:)) - TP;

    prec(u) = TP/(TP+FP+eps);
    rec(u)  = TP/(TP+FN+eps);
    f1(u)   = 2 * prec(u) * rec(u) / (prec(u)+rec(u)+eps);
end

macroPrec = mean(prec);
macroRec = mean(rec);
macroF1 = mean(f1);

fprintf("\n======= FFNN METRICS =======\n");
fprintf("Macro Precision: %.2f%%\n", macroPrec*100);
fprintf("Macro Recall:    %.2f%%\n", macroRec*100);
fprintf("Macro F1 Score:  %.2f%%\n", macroF1*100);


%% ===========================
%%  AUTHENTICATION METRICS
%%  FAR / FRR / EER PER-USER
%% ===========================
users = unique(yte);
nUsers = numel(users);
taus = 0:0.001:1;

FAR = zeros(nUsers,length(taus));
FRR = zeros(nUsers,length(taus));
EER_user = zeros(nUsers,1);
tau_at_eer = zeros(nUsers,1);

fprintf("\n====== USER AUTHENTICATION (FFNN) ======\n");

for ui = 1:nUsers
    u = users(ui);
    genuine = scores_nn(yte==u, u);
    impostor = scores_nn(yte~=u, u);

    for ti = 1:length(taus)
        t = taus(ti);
        FAR(ui,ti) = mean(impostor >= t);
        FRR(ui,ti) = mean(genuine < t);
    end

    % Find EER
    [~, idx] = min(abs(FAR(ui,:) - FRR(ui,:)));
    EER_user(ui) = mean([FAR(ui,idx), FRR(ui,idx)]);
    tau_at_eer(ui) = taus(idx);

    fprintf("User %d → EER = %.2f%%   (tau = %.3f)\n", u, EER_user(ui)*100, tau_at_eer(ui));
end

EER_mean = mean(EER_user);
fprintf("\nAverage System EER: %.2f%%\n", EER_mean*100);


%% ===========================
%% SAVE METRICS
%% ===========================
save(fullfile(resultsFolder,"metrics.mat"), ...
    "acc_knn","acc_svm","acc_nn", ...
    "macroPrec","macroRec","macroF1", ...
    "EER_user","EER_mean","tau_at_eer");

fprintf("Saved results → %s\n", fullfile(resultsFolder,"metrics.mat"));
fprintf("\n=== Evaluation complete ===\n");
