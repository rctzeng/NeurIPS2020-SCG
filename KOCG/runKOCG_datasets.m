%%%%%% Real-World Dataset %%%%%%
DATASET_LIST = ["wow8", "bitcoin", "wikivot", "referendum", "slashdot", "wikicon", "epinions", "wikipol"];

% K=2
for name = DATASET_LIST
    fprintf('============================== Running %s ==============================\n', name);
    load(sprintf('../datasets/%s.mat', name));
    A = (A + A.')/2; % required by KOCG(KDD'16)
    [X_enumKOCG_cell, time] = enumKOCG(A, [], 2, 1, 50); % beta=50 is the default setting
    X = X_enumKOCG_cell{1};
    save(sprintf('K2/result_%s_p%d', name, 1), 'X');
    for i = 2:min(5000,length(X_enumKOCG_cell))
        X = X + X_enumKOCG_cell{i};
        save(sprintf('K2/result_%s_p%d', name, i), 'X', '-v6');
    end
end

% K=6
for name = DATASET_LIST
    fprintf('============================== Running %s ==============================\n', name);
    load(sprintf('../datasets/%s.mat', name));
    A = (A + A.')/2; % required by KOCG(KDD'16)
    [X_enumKOCG_cell, time] = enumKOCG(A, [], 6, 1.0/5, 50); % beta=50 is the default setting
    X = X_enumKOCG_cell{1};
    save(sprintf('K6/result_%s_p%d', name, 1), 'X');
    for i = 2:min(5000,length(X_enumKOCG_cell))
        X = X + X_enumKOCG_cell{i};
        save(sprintf('K6/result_%s_p%d', name, i), 'X', '-v6');
    end
end
