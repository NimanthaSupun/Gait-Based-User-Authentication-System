% 02_preprocess_features.m
% Per-window preprocessing and feature extraction
% Produces features matrix X (nWindows x nFeatures)

load('config.mat','params');
load('windows_raw.mat','allWindows','allUsers','allSess','filenames');

nWindows = numel(allWindows);
X = zeros(nWindows, 74);  % prealloc estimate (we'll keep 74 then remove const cols)
for i = 1:nWindows
    W = allWindows{i}; % window_len x 6
    ax = W(:,1); ay = W(:,2); az = W(:,3);
    gx = W(:,4); gy = W(:,5); gz = W(:,6);

    % -------------------------
    % preprocessing (per-window)
    % -------------------------
    % 1) Remove DC offset
    ax = ax - mean(ax); ay = ay - mean(ay); az = az - mean(az);
    gx = gx - mean(gx); gy = gy - mean(gy); gz = gz - mean(gz);

    % 2) Smooth (moving average)
    ax = movmean(ax,5); ay = movmean(ay,5); az = movmean(az,5);
    gx = movmean(gx,5); gy = movmean(gy,5); gz = movmean(gz,5);

    % 3) Replace outliers
    ax = filloutliers(ax,'linear'); ay = filloutliers(ay,'linear'); az = filloutliers(az,'linear');
    gx = filloutliers(gx,'linear'); gy = filloutliers(gy,'linear'); gz = filloutliers(gz,'linear');

    % 4) Per-window z-score (stabilizes features)
    ax = (ax - mean(ax)) / (std(ax)+eps);
    ay = (ay - mean(ay)) / (std(ay)+eps);
    az = (az - mean(az)) / (std(az)+eps);
    gx = (gx - mean(gx)) / (std(gx)+eps);
    gy = (gy - mean(gy)) / (std(gy)+eps);
    gz = (gz - mean(gz)) / (std(gz)+eps);

    % -------------------------
    % features
    % -------------------------
    td = @(v)[ mean(v), std(v), skewness(v), kurtosis(v), min(v), max(v), iqr(v), sum(v.^2) ];
    Ft_td = [td(ax), td(ay), td(az), td(gx), td(gy), td(gz)]; % 48

    % magnitudes
    amag = sqrt(ax.^2 + ay.^2 + az.^2);
    gmag = sqrt(gx.^2 + gy.^2 + gz.^2);
    Ft_mag = [td(amag), td(gmag)]; %16

    % correlations (accelerometer)
    cxy = corr(ax,ay,'Rows','complete'); cxz = corr(ax,az,'Rows','complete'); cyz = corr(ay,az,'Rows','complete');
    Ft_corr = [cxy, cxz, cyz]; %3

    % frequency features on accel magnitude
    Nf = numel(amag);
    Y = abs(fft(amag));
    P = Y(1:floor(Nf/2));
    freqBins = (0:numel(P)-1);
    [~,domI] = max(P);
    domFreq = domI;
    centroid = sum(freqBins'.*P)/sum(P+eps);
    spread = var(P);
    Pmean = mean(P); Pstd = std(P);
    pnorm = P./(sum(P)+eps);
    specEnt = -sum(pnorm.*log2(pnorm+eps));
    rollIdx = find(cumsum(P) >= 0.85*sum(P),1,'first'); if isempty(rollIdx), rollIdx = numel(P); end
    Ft_fd = [domFreq, centroid, spread, Pmean, Pstd, specEnt, rollIdx]; %7

    row = [Ft_td, Ft_mag, Ft_corr, Ft_fd];
    X(i,1:numel(row)) = row;
end

% Save raw features
if params.saveMat
    save('features_raw.mat','X','allUsers','allSess','filenames','-v7.3');
end
fprintf("Extracted raw features: %d x %d\n", size(X,1), size(X,2));
