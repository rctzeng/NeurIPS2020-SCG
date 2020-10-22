% K=6
for t = 1:20
    for p = 1:7
        fprintf('============================== Running sbm_t%d_p%d ==============================\n', t-1, p-1);
        load(sprintf('../datasets/sbm_t%d_p%d.mat', t-1, p-1));
        A = (A + A.')/2; % required by KOCG(KDD'16)
        [X_enumKOCG_cell, time] = enumKOCG(A, [], 6, 1.0/5, 50); % beta=50 is the default setting
        X = X_enumKOCG_cell{1};
        save(sprintf('K6/sbm_t%d_p%d_p%d', t-1, p-1, 1), 'X', '-v6');
        for i = 2:min(100,length(X_enumKOCG_cell))
            X = X + X_enumKOCG_cell{i};
            save(sprintf('K6/sbm_t%d_p%d_p%d', t-1, p-1, i), 'X', '-v6');
        end
    end
end
