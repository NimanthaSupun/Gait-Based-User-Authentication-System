%% ============================================================
%  generate_visualizations.m
%  Generate ALL figures and save to /results/ folder
% ============================================================

clear; clc; close all;

%% Create results directory
outdir = fullfile(pwd, "results");
if ~exist(outdir, 'dir')
    mkdir(outdir);
end
disp("Saving all visualizations to: " + outdir);


%% ================================================
%  Load feature matrix & labels
% ================================================
load dataset.mat   % Contains Xtr, ytr, Xte, yte

% Combine train and test for visualization
X_windows = [Xtr; Xte];
y_windows = [ytr; yte];


%% ================================================
%  1. PCA Visualization (2D & 3D)
% ================================================
[coeff, score, latent, tsq, explained] = pca(X_windows);

% -------- 2D PCA --------
figure;
gscatter(score(:,1), score(:,2), y_windows);
title('2D PCA of Gait Features');
xlabel(['PC1 (' num2str(explained(1),'%.1f') '%)']);
ylabel(['PC2 (' num2str(explained(2),'%.1f') '%)']);
grid on;

saveas(gcf, fullfile(outdir, "pca_2d.png"));

% -------- 3D PCA --------
figure;
scatter3(score(:,1), score(:,2), score(:,3), 18, y_windows, 'filled');
title('3D PCA Visualization of Gait Features');
xlabel(['PC1 (' num2str(explained(1),'%.1f') '%)']);
ylabel(['PC2 (' num2str(explained(2),'%.1f') '%)']);
zlabel(['PC3 (' num2str(explained(3),'%.1f') '%)']);
colorbar; colormap jet; grid on; rotate3d on;

saveas(gcf, fullfile(outdir, "pca_3d.png"));

disp("✓ PCA plots saved.");



%% ================================================
%  2. Feature Correlation Heatmap
% ================================================
corrMat = corr(X_windows, 'rows','complete');

figure;
imagesc(corrMat);
title('Feature Correlation Matrix');
xlabel('Feature Index'); ylabel('Feature Index');
colorbar; colormap jet;

saveas(gcf, fullfile(outdir, "correlation_heatmap.png"));
disp("✓ Correlation heatmap saved.");



%% ================================================
%  3. Intra-Class vs Inter-Class Distances
% ================================================
users = unique(y_windows);
intra = []; inter = [];

for u = users'
    idx = (y_windows == u);
    X_u = X_windows(idx,:);
    X_non = X_windows(~idx,:);
    
    % Intra-user pairwise distances
    intra = [intra; pdist(X_u,'euclidean')];
    
    % Inter-user distances
    for i = 1:size(X_u,1)
        d = sqrt(sum((X_non - X_u(i,:)).^2, 2));
        inter = [inter; d];
    end
end

figure;
histogram(intra, 40, 'Normalization', 'pdf'); hold on;
histogram(inter, 40, 'Normalization', 'pdf');
legend('Intra-Class Distances','Inter-Class Distances');
title('Intra vs Inter User Feature Distances');
xlabel('Euclidean Distance'); ylabel('Density');
grid on;

saveas(gcf, fullfile(outdir, "distance_distribution.png"));
disp("✓ Distance distribution plot saved.");



%% ================================================
%  4. Feature Importance (ReliefF Ranking)
% ================================================
[idx, weights] = relieff(X_windows, y_windows, 50);

figure;
bar(weights(idx), 'FaceColor',[0 .45 .74]);
title('Feature Importance Ranking (ReliefF)');
xlabel('Ranked Feature'); ylabel('Importance Weight');
grid on;

saveas(gcf, fullfile(outdir, "feature_importance.png"));
disp("✓ Feature importance plot saved.");



%% ================================================
%  5. Model Accuracy Comparison
% ================================================
load metrics.mat  % Contains acc_knn, acc_svm, acc_ffnn

models = categorical({'KNN','SVM','FFNN'});
accs = [acc_knn, acc_svm, acc_ffnn];

figure;
bar(models, accs, 'FaceColor',[0 .5 .8]);
ylim([0 100]); ylabel('Accuracy (%)');
title('Model Performance Comparison');
grid on;

saveas(gcf, fullfile(outdir, "accuracy_comparison.png"));
disp("✓ Model accuracy comparison saved.");



%% ================================================
%  6. Per-Class Precision & Recall
% ================================================
load metrics.mat  % Contains precision_class, recall_class

classes = categorical(string(1:10));

figure;
subplot(2,1,1);
bar(classes, precision_class*100, 'FaceColor',[0 .6 .4]);
title('Per-Class Precision (%)'); ylim([0 100]);
ylabel('%'); grid on;

subplot(2,1,2);
bar(classes, recall_class*100, 'FaceColor',[0 .4 .6]);
title('Per-Class Recall (%)'); ylim([0 100]);
ylabel('%'); xlabel('User ID'); grid on;

saveas(gcf, fullfile(outdir, "precision_recall_per_class.png"));
disp("✓ Precision/Recall charts saved.");



%% ================================================
%  DONE
% ================================================
disp("===============================================");
disp(" ALL VISUALIZATION DIAGRAMS SUCCESSFULLY SAVED ");
disp(" Location: " + outdir);
disp("===============================================");
