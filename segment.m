% 01_segment.m
% Segment all CSV files into overlapping windows and label them (user, session)

load('config.mat','dataFolder','params');

files = dir(fullfile(dataFolder,'*.csv'));
if isempty(files)
    error("No CSV files in %s", dataFolder);
end

allWindows = {}; allUsers = []; allSess = []; filenames = {};
for f = 1:numel(files)
    fname = files(f).name;
    filepath = fullfile(files(f).folder, fname);
    M = readmatrix(filepath);
    if size(M,2) < 6
        error("File %s has fewer than 6 columns (expected 6).", fname);
    end
    M = M(:,1:6); % only first 6 columns
    N = size(M,1);
    if N < params.minSamplesPerFile
        warning("File %s has only %d samples (<%d).", fname, N, params.minSamplesPerFile);
    end

    % session
    if contains(fname,'FD','IgnoreCase',true)
        sess = 1;
    elseif contains(fname,'MD','IgnoreCase',true)
        sess = 2;
    else
        error("Filename %s must contain FD or MD.", fname);
    end

    % user id
    t = regexp(fname,'U(\d+)','tokens','once');
    if isempty(t), error("Filename %s must include U<id>.", fname); end
    uid = str2double(t{1});

    % segment
    idx = 1:params.hop:(N - params.window_len + 1);
    for k = 1:numel(idx)
        w = M(idx(k):(idx(k)+params.window_len-1), :);
        allWindows{end+1} = w; %#ok<SAGROW>
        allUsers(end+1) = uid; %#ok<SAGROW>
        allSess(end+1) = sess; %#ok<SAGROW>
        filenames{end+1} = fname; %#ok<SAGROW>
    end
end

nWindows = numel(allWindows);
fprintf("Segmented %d windows from %d files.\n", nWindows, numel(files));
if params.saveMat
    save('windows_raw.mat','allWindows','allUsers','allSess','filenames','-v7.3');
end
