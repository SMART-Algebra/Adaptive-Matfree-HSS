function [node, k_A, table_rows, t_fact_no_mv, t_mv, n_mv, diagnostics] = factorize_tree_2D( ...
        method, num_levels, pt_leaf, fixed_pt_leaf, n1d, ...
        kernel_type, symmetric_kernel_list, ...
        spd_kernel_list, user_tolerance, step_b, ...
        mv, mvT, ...
        Y_leaf, Omega_leaf, collect_diagnostics, comparison_seed, sample_distribution, perform_rank_pruning)

    if nargin < 15
        collect_diagnostics = false;
    end
    if nargin < 16
        comparison_seed = [];
    end
    if nargin < 17 || isempty(sample_distribution)
        sample_distribution = 'real';
    end
    if nargin < 18 || isempty(perform_rank_pruning)
        perform_rank_pruning = 'leaf';
    end

    level_stats = struct([]);

    % ------------------- Dimensions -------------------
    leaf_side = sqrt(fixed_pt_leaf);
    n         = n1d^2;
    
    % ------------------- Symmetry ---------------------
    if ismember(kernel_type, symmetric_kernel_list)
        is_unsymmetric = false;
    else
        is_unsymmetric = true;
    end
    
    % ------------------- Nodes ------------------------
    nodes_per_level = 4.^(0:num_levels-1);
    num_nodes       = sum(nodes_per_level);
    
    if is_unsymmetric
        nodeStruct = struct('U',[],'V',[],'X_rr',[],'X_rs',[], 'X_srX_rr', [], 'X_rsX_rr', [], ...
                            'Y_tilde',[],'Omega_tilde',[], 'Y_tilde_n',[],'Omega_tilde_n',[], ...
                            'isLeaf',false, ...
                            'Indices', struct('original', [], 'skeleton', [], 'redundant', []));
    else
        nodeStruct = struct('U', [], 'X_rr', [], 'X_rs', [], 'X_srX_rr', [], ...
                            'Y_tilde', [], 'Omega_tilde', [], ...
                            'isLeaf', false, 'rank', [], ...
                            'Indices', struct('original', [], 'skeleton', [], 'redundant', []));
    end
    node = repmat(nodeStruct, num_nodes, 1);
    
    % ------------------- Timing -----------------------
    add_cols_num('reset');
    add_cols_time('reset');
    t_fact_tic = tic;
    
    % ------------------- Leaf level -------------------
    fprintf("CURRENT LEVEL: %d ----------------\n", num_levels);
    
    start_cols_Omega_all   = zeros(num_levels, 1);
    num_blocks_side        = 2^(num_levels - 1);
    pts_per_block          = leaf_side;     % along one side (leaf only)
    start_cols_Omega       = fixed_pt_leaf; % initial columns available at leaves
    start_cols_Omega_all(1)= start_cols_Omega;
    
    if is_unsymmetric
        % Prepare the adjoint sample via A^*.
        Y_leaf_n = zeros(n, pt_leaf);
        for j = 1:pt_leaf
            Y_leaf_n(:, j) = mvT(Omega_leaf(:, j));
        end
    
        reset_comparison_rng(comparison_seed, num_levels, 'U');
        [U_b_i_all, Omega_leaf_final, Y_leaf_final, k_leaf, num_iterations, stats_u] = ...
            building_bases(method, user_tolerance, step_b, mv, n, Y_leaf, Omega_leaf, ...
                num_blocks_side, pts_per_block, fixed_pt_leaf, node, num_levels, num_levels, 'UV', symmetric_kernel_list, 2, start_cols_Omega, collect_diagnostics, sample_distribution, perform_rank_pruning);
        level_stats = append_basis_stat(level_stats, stats_u, 'U');
    
        reset_comparison_rng(comparison_seed, num_levels, 'V');
        [V_b_i_all, Omega_leaf_final_n, Y_leaf_final_n, k_leaf_n, num_iterations_n, stats_v] = ...
            building_bases(method, user_tolerance, step_b, mvT, n, Y_leaf_n, Omega_leaf, ...
                num_blocks_side, pts_per_block, fixed_pt_leaf, node, num_levels, num_levels, 'VU', symmetric_kernel_list, 2, start_cols_Omega, collect_diagnostics, sample_distribution, perform_rank_pruning);
        level_stats = append_basis_stat(level_stats, stats_v, 'V');
    
        k_leaf        = max(k_leaf, k_leaf_n);
        num_iterations= max(num_iterations, num_iterations_n);
    
        % Orthonormalize V blocks
        V_full_leaf = cell(size(V_b_i_all));
        for idx = 1:length(V_b_i_all)
            V_full_leaf{idx} = wy_or_identity(V_b_i_all{idx});
        end
    else
        reset_comparison_rng(comparison_seed, num_levels, 'U');
        [U_b_i_all, Omega_leaf_final, Y_leaf_final, k_leaf, num_iterations, stats_u] = ...
            building_bases(method, user_tolerance, step_b, mv, n, Y_leaf, Omega_leaf, ...
                num_blocks_side, pts_per_block, fixed_pt_leaf, node, num_levels, num_levels, kernel_type, symmetric_kernel_list, 2, start_cols_Omega, collect_diagnostics, sample_distribution, perform_rank_pruning);
        level_stats = append_basis_stat(level_stats, stats_u, 'U');
    end
    
    % Ensure enough columns for the parent (need 4 * k_leaf)
    required_cols_parent = 4 * k_leaf;
    current_cols         = size(Omega_leaf_final, 2);
    while current_cols < required_cols_parent
        [new_cols_Omega, new_cols_Y] = add_columns_at_leaves(step_b, mv, n, sample_distribution);
        Omega_leaf_final = [Omega_leaf_final, new_cols_Omega];
        Y_leaf_final     = [Y_leaf_final, new_cols_Y];
        current_cols     = size(Omega_leaf_final, 2);
    end

    if is_unsymmetric
        current_cols_n = size(Omega_leaf_final_n, 2);
        while current_cols_n < required_cols_parent
            [new_cols_Omega_n, new_cols_Y_n] = add_columns_at_leaves(step_b, mvT, n, sample_distribution);
            Omega_leaf_final_n = [Omega_leaf_final_n, new_cols_Omega_n];
            Y_leaf_final_n     = [Y_leaf_final_n,     new_cols_Y_n];
            current_cols_n     = size(Omega_leaf_final_n, 2);
        end
    end
    
    % Orthonormalize U blocks
    U_full_leaf = cell(size(U_b_i_all));
    for idx = 1:length(U_b_i_all)
        U_full_leaf{idx} = wy_or_identity(U_b_i_all{idx});
    end
    
    % Table rows (leaf row first)
    table_rows = {};
    table_rows(end+1, :) = {sprintf('%g', user_tolerance), num_levels, n, num_blocks_side, start_cols_Omega, k_leaf, '', '', '', '', '', ''};
    
    % Store leaf results
    if ~is_unsymmetric
        node = store_node_results(node, Omega_leaf_final, Y_leaf_final, U_full_leaf, pts_per_block, num_levels, num_levels, k_leaf, ...
            [], [], [], kernel_type, symmetric_kernel_list, spd_kernel_list, 2, n1d);
    else
        node = store_node_results(node, Omega_leaf_final, Y_leaf_final, U_full_leaf, pts_per_block, num_levels, num_levels, k_leaf, ...
            Omega_leaf_final_n, Y_leaf_final_n, V_full_leaf, kernel_type, symmetric_kernel_list, spd_kernel_list, 2, n1d);
    end
    
    % Cleanup leaf temporaries (keep node)
    clear Y_leaf Y_leaf_n;
    
    clear U_b_i_all U_full_leaf V_b_i_all V_full_leaf ...
          Omega_leaf_final Y_leaf_final Omega_leaf_final_n Y_leaf_final_n;
    
    % ---------------- Internal (non-leaf) levels ----------------
    k_A = k_leaf;
    start_cols_Omega = 4 * k_A;
    start_cols_Omega_all(2) = start_cols_Omega;
    row_ix = 3;
    
    for current_level = num_levels-1:-1:2
        fprintf("CURRENT LEVEL: %d ----------------\n", current_level);
    
        num_blocks_side = num_blocks_side / 2;
        pts_per_block   = pts_per_block * 2;
    
        if is_unsymmetric
            [Y_current_all, Omega_current_all, Y_current_all_n, Omega_current_all_n] = ...
                get_YOmega_non_leaf(node, current_level, kernel_type, symmetric_kernel_list, 2);
    
            reset_comparison_rng(comparison_seed, current_level, 'U');
            [U_b_i_all, Omega_to_use, Y_to_use, k_A, num_iterations, stats_u] = ...
                building_bases(method, user_tolerance, step_b, mv, n, Y_current_all, Omega_current_all, ...
                    num_blocks_side, pts_per_block, fixed_pt_leaf, node, current_level, num_levels, 'UV', symmetric_kernel_list, 2, start_cols_Omega, collect_diagnostics, sample_distribution, perform_rank_pruning);
            level_stats = append_basis_stat(level_stats, stats_u, 'U');
    
            reset_comparison_rng(comparison_seed, current_level, 'V');
            [V_b_i_all, Omega_to_use_n, Y_to_use_n, k_A_n, num_iterations_n, stats_v] = ...
                building_bases(method, user_tolerance, step_b, mvT, n, Y_current_all_n, Omega_current_all_n, ...
                    num_blocks_side, pts_per_block, fixed_pt_leaf, node, current_level, num_levels, 'VU', symmetric_kernel_list, 2, start_cols_Omega, collect_diagnostics, sample_distribution, perform_rank_pruning);
            level_stats = append_basis_stat(level_stats, stats_v, 'V');
    
            k_A = max(k_A, k_A_n);
            num_iterations = max(num_iterations, num_iterations_n);
    
            % Orthonormalize V
            V_full = cell(size(V_b_i_all));
            for idx = 1:length(V_b_i_all)
                V_full{idx} = wy_or_identity(V_b_i_all{idx});
            end
        else
            [Y_current_all, Omega_current_all] = ...
                get_YOmega_non_leaf(node, current_level, kernel_type, symmetric_kernel_list, 2);
    
            reset_comparison_rng(comparison_seed, current_level, 'U');
            [U_b_i_all, Omega_to_use, Y_to_use, k_A, num_iterations, stats_u] = ...
                building_bases(method, user_tolerance, step_b, mv, n, Y_current_all, Omega_current_all, ...
                    num_blocks_side, pts_per_block, fixed_pt_leaf, node, current_level, num_levels, kernel_type, symmetric_kernel_list, 2, start_cols_Omega, collect_diagnostics, sample_distribution, perform_rank_pruning);
            level_stats = append_basis_stat(level_stats, stats_u, 'U');
        end
    
        % Ensure enough starting columns for parent: start_cols_Omega = 4 * k_A
        start_cols_Omega = 4 * k_A;
        if current_level == 2
            % The parent is the root, whose block A^(0) is recovered from
            % these samples alone (no further rank search adds any). With
            % exactly 4 k_A samples that recovery is a square system: the
            % cavity rows at eps = 1e-12 in Table 7 (N = 256^2, 512^2) have
            % n_mv = |A^(0)| = 4608 and 9984 for this reason, and a square
            % Gaussian system has no averaging and a condition number that
            % grows with its dimension. Measured on 2026-10-10 (seed 0): the
            % cavity at 128^2, eps = 1e-12, square on the laptop, went from
            % relres 2.1e-10 to 4.8e-12 with a 10% margin; 3D Poisson 64^2,
            % 1e-9 (ratio 1.014) from 1.1e-8 to 4.9e-9; rows with a ratio
            % above about 1.1 barely move. The root is therefore sampled to
            % at least ROOT_OVERSAMPLE times its dimension, rounded up to
            % whole blocks of step_b. The global HSS_ROOT_OVERSAMPLE overrides
            % the default (1 restores the behavior before 2026-10-10).
            global HSS_ROOT_OVERSAMPLE %#ok<GVMIS,TLEV>
            root_oversample = 1.1;
            if ~isempty(HSS_ROOT_OVERSAMPLE) && isscalar(HSS_ROOT_OVERSAMPLE) && HSS_ROOT_OVERSAMPLE >= 1
                root_oversample = HSS_ROOT_OVERSAMPLE;
            end
            start_cols_Omega = max(4 * k_A, step_b * ceil(root_oversample * 4 * k_A / step_b));
        end
        start_cols_Omega_all(row_ix) = start_cols_Omega;
    
        if ~is_unsymmetric
            current_cols = size(Omega_to_use, 2);
            while current_cols < start_cols_Omega
                [Y_to_use_all, Omega_to_use_all] = ...
                    expand_Omega_Y(node, current_level, step_b, mv, n, fixed_pt_leaf, kernel_type, symmetric_kernel_list, 2, sample_distribution);
                Omega_to_use = [Omega_to_use, Omega_to_use_all];
                Y_to_use     = [Y_to_use,     Y_to_use_all];
                current_cols = size(Omega_to_use, 2);
            end
        else
            current_cols = size(Omega_to_use, 2);
            while current_cols < start_cols_Omega
                [Y_to_use_all, Omega_to_use_all] = ...
                    expand_Omega_Y(node, current_level, step_b, mv, n, fixed_pt_leaf, 'UV', symmetric_kernel_list, 2, sample_distribution);
                Omega_to_use = [Omega_to_use, Omega_to_use_all];
                Y_to_use     = [Y_to_use,     Y_to_use_all];
                current_cols = size(Omega_to_use, 2);
            end
    
            current_cols_n = size(Omega_to_use_n, 2);
            while current_cols_n < start_cols_Omega
                [Y_to_use_all_n, Omega_to_use_all_n] = ...
                    expand_Omega_Y(node, current_level, step_b, mvT, n, fixed_pt_leaf, 'VU', symmetric_kernel_list, 2, sample_distribution);
                Omega_to_use_n = [Omega_to_use_n, Omega_to_use_all_n];
                Y_to_use_n     = [Y_to_use_n,     Y_to_use_all_n];
                current_cols_n = size(Omega_to_use_n, 2);
            end
        end
    
        % Orthonormalize U
        U_full = cell(size(U_b_i_all));
        for idx = 1:length(U_b_i_all)
            U_full{idx} = wy_or_identity(U_b_i_all{idx});
        end
    
        % Append per-level table row
        n_current = start_cols_Omega_all(row_ix-1) * num_blocks_side^2;
        table_rows(end+1, :) = {'', current_level, n_current, num_blocks_side, start_cols_Omega_all(row_ix-1), k_A, '', '', '', '', '', ''};
    
        % Store nodes for this level
        if ~is_unsymmetric
            node = store_node_results(node, Omega_to_use, Y_to_use, U_full, pts_per_block, current_level, num_levels, k_A, ...
                [], [], [], kernel_type, symmetric_kernel_list, spd_kernel_list, 2, n1d);
        else
            node = store_node_results(node, Omega_to_use, Y_to_use, U_full, pts_per_block, current_level, num_levels, k_A, ...
                Omega_to_use_n, Y_to_use_n, V_full, kernel_type, symmetric_kernel_list, spd_kernel_list, 2, n1d);
        end
    
        % Cleanup
        clear Y_current_all Omega_current_all Y_current_all_n Omega_current_all_n
        clear U_full V_full U_b_i_all V_b_i_all Omega_to_use Omega_to_use_n Y_to_use Y_to_use_n;
    
        row_ix = row_ix + 1;
    end
    
    % --------------------- Root ----------------------
    nd = 1;
    child_indices = 2:5;         % quadtree: 4 children
    
    node(nd).isLeaf = false;
    
    [Y_i_hat_3_hat_3, Omega_i_hat_3_hat_3] = combine_children_results(node(child_indices), kernel_type, symmetric_kernel_list, []);
    
    original_indices = [];
    for ii = 1:length(child_indices)
        cid = child_indices(ii);
        original_indices = [original_indices, node(cid).Indices.skeleton]; % not sorted
    end
    
    node(nd).Indices.original  = original_indices;
    node(nd).Indices.redundant = original_indices;
    
    lambda = 1e-13;
    Y_i_hat_3_hat_3_T = Y_i_hat_3_hat_3';
    n_cols = size(Omega_i_hat_3_hat_3, 1);
    X_rr_trans = [Omega_i_hat_3_hat_3'; lambda*eye(n_cols)] \ [Y_i_hat_3_hat_3_T; zeros(n_cols, size(Y_i_hat_3_hat_3_T, 2))];

    X_rr_raw = X_rr_trans';

    if ismember(kernel_type, spd_kernel_list)
        % 1. Force Symmetry
        X_rr_i = 0.5 * (X_rr_raw + X_rr_raw.');
        
        % 2. Force Positive Definiteness (The Loop)
        [~, p] = chol(X_rr_i);
        if p > 0
            matrix_norm = norm(X_rr_i, 'fro');
            shift = 1e-12 * matrix_norm;
            if shift == 0, shift = 1e-12; end
            while p > 0
                X_rr_i = X_rr_i + shift * eye(size(X_rr_i));
                [~, p] = chol(X_rr_i);
                shift = shift * 2;
            end
        end
        node(nd).X_rr = X_rr_i;
    
    elseif ismember(kernel_type, symmetric_kernel_list)
        % Just Symmetry
        node(nd).X_rr = 0.5 * (X_rr_raw + X_rr_raw.');
    else
        % Unsymmetric
        node(nd).X_rr = X_rr_raw;
    end

    clear Y_i_hat_3_hat_3 Omega_i_hat_3_hat_3 ...
        Y_i_hat_3_hat_3_T X_rr_trans X_rr_raw original_indices;
    node = discard_sampling_state(node);
    
    % ------------------- Timings ------------------
    t_mv_internal  = add_cols_time('get');
    t_mv           = round(t_mv_internal, 2);
    
    diagnostics = summarize_basis_stats(method, level_stats);
    t_fact_no_mv   = toc(t_fact_tic) - t_mv_internal - diagnostics.diagnostics_time;
    t_fact_no_mv   = round(t_fact_no_mv, 2);
    
    % ------------------- Number of matvecs ------------------
    n_mv_internal  = add_cols_num('get');
    n_mv           = n_mv_internal + size(Omega_leaf, 2);
    
    add_cols_time('reset');
    add_cols_num('reset');
    
    fprintf("#######################################################################################\n");
end
