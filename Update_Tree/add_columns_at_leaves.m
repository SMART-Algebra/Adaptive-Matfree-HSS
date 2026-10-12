function [new_cols_Omega_leaf, new_cols_Y_leaf] = add_columns_at_leaves(step_b, mv, n, sample_distribution)
    % addColumnsAtLeaves: Adds step_b columns to Omega and computes Y=A*Omega
    % on leaf level using toeplitzMatVec rather than forming A.

    if nargin < 4 || isempty(sample_distribution)
        sample_distribution = 'real';
    end

    t0 = tic;

    new_cols_Omega_leaf = draw_gaussian(n, step_b, sample_distribution);
    % Match the sketch type up front. This avoids reallocating the sample
    % matrix when a complex operator is paired with circular complex tests.
    new_cols_Y_leaf = zeros(n, step_b, 'like', new_cols_Omega_leaf);

    for j = 1:step_b
        new_cols_Y_leaf(:,j) = mv(new_cols_Omega_leaf(:,j));
    end

    add_cols_num('add', step_b);    % accumulate total matvec count
    add_cols_time('add', toc(t0));  % accumulate elapsed seconds
                                  
end
