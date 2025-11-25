% 06_user_thresholds.m
% Compute per-user thresholds (e.g., at EER) and evaluate new FAR/FRR

load('metrics.mat','EER_user','tau_at_eer');
load('dataset.mat','Xte','yte');
load('models.mat','net');

scores = net(Xte')';
users = unique(yte);
nUsers = numel(users);

newFAR = zeros(nUsers,1); newFRR = zeros(nUsers,1);
for ui=1:nUsers
    u = users(ui);
    tau = tau_at_eer(ui);
    genuine = scores(yte==u,u);
    impostor = scores(yte~=u,u);
    newFAR(ui) = mean(impostor >= tau);
    newFRR(ui) = mean(genuine < tau);
    fprintf("User %d: tau=%.3f FAR=%.3f FRR=%.3f\n", u, tau, newFAR(ui), newFRR(ui));
end
fprintf("Avg FAR after user-specific tau: %.3f Avg FRR: %.3f\n", mean(newFAR), mean(newFRR));

if params.saveMat
    save('user_thresholds.mat','tau_at_eer','newFAR','newFRR','-v7.3');
end
