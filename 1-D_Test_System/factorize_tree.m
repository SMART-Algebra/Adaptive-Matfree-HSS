function [node, k_A, table_rows, t_fact_no_mv, t_mv, n_mv, diagnostics] = factorize_tree( ...
        method, num_levels, pt_leaf, fixed_pt_leaf, n, ...
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

    % Decide symmetry
    is_unsymmetric = ~ismember(kernel_type, symmetric_kernel_list);
    
    % Pre-allocate node array
    nodes_per_level = 2.^(0:num_levels-1);
    num_nodes = sum(nodes_per_level);
    
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
    
    % Timers
    add_cols_num('reset');
    add_cols_time('reset');
    t_fact_tic = tic;
    
    % ---------------- Leaf level ----------------
    fprintf("CURRENT LEVEL: %d ----------------\n", num_levels);
    num_blocks = n / pt_leaf;
    
    if is_unsymmetric
        % Build the adjoint sample using A^*.
        Y_leaf_n = zeros(n, pt_leaf);
        for j = 1:pt_leaf
            Y_leaf_n(:, j) = mvT(Omega_leaf(:, j));
        end
    
        reset_comparison_rng(comparison_seed, num_levels, 'U');
        [U_b_i_all, Omega_leaf_final, Y_leaf_final, k_leaf, num_iterations, stats_u] = ...
            building_bases(method, user_tolerance, step_b, mv, n, Y_leaf, Omega_leaf, ...
                num_blocks, pt_leaf, fixed_pt_leaf, node, num_levels, num_levels, 'UV', symmetric_kernel_list, 1, [], collect_diagnostics, sample_distribution, perform_rank_pruning);
        level_stats = append_basis_stat(level_stats, stats_u, 'U');
    
        reset_comparison_rng(comparison_seed, num_levels, 'V');
        [V_b_i_all, Omega_leaf_final_n, Y_leaf_final_n, k_leaf_n, num_iterations_n, stats_v] = ...
            building_bases(method, user_tolerance, step_b, mvT, n, Y_leaf_n, Omega_leaf, ...
                num_blocks, pt_leaf, fixed_pt_leaf, node, num_levels, num_levels, 'VU', symmetric_kernel_list, 1, [], collect_diagnostics, sample_distribution, perform_rank_pruning);
        level_stats = append_basis_stat(level_stats, stats_v, 'V');
    
        k_leaf = max(k_leaf, k_leaf_n);
        num_iterations = max(num_iterations, num_iterations_n);
    
        % Square/orthonormalize V
        V_full_leaf = cell(size(V_b_i_all));
        for idx = 1:length(V_b_i_all)
            V_full_leaf{idx} = wy_or_identity(V_b_i_all{idx});
        end
    else
        reset_comparison_rng(comparison_seed, num_levels, 'U');
        [U_b_i_all, Omega_leaf_final, Y_leaf_final, k_leaf, num_iterations, stats_u] = ...
            building_bases(method, user_tolerance, step_b, mv, n, Y_leaf, Omega_leaf, ...
                num_blocks, pt_leaf, fixed_pt_leaf, node, num_levels, num_levels, kernel_type, symmetric_kernel_list, 1, [], collect_diagnostics, sample_distribution, perform_rank_pruning);
        level_stats = append_basis_stat(level_stats, stats_u, 'U');
    end
    
    % Square/orthonormalize U
    U_full_leaf = cell(size(U_b_i_all));
    for idx = 1:length(U_b_i_all)
        U_full_leaf{idx} = wy_or_identity(U_b_i_all{idx});
    end
    
    % Table rows (leaf row first)
    table_rows = {};
    table_rows(end+1, :) = {sprintf('%g', user_tolerance), num_levels, n, num_blocks, pt_leaf, k_leaf, '', '', '', '', '', '', ''};
    
    % Store leaf results
    if is_unsymmetric
        node = store_node_results(node, Omega_leaf_final, Y_leaf_final, U_full_leaf, pt_leaf, num_levels, num_levels, k_leaf, ...
                    Omega_leaf_final_n, Y_leaf_final_n, V_full_leaf, kernel_type, symmetric_kernel_list, spd_kernel_list);
    else
        node = store_node_results(node, Omega_leaf_final, Y_leaf_final, U_full_leaf, pt_leaf, num_levels, num_levels, k_leaf, ...
                    [], [], [], kernel_type, symmetric_kernel_list, spd_kernel_list);
    end
    
    clear U_b_i_all U_full_leaf Omega_leaf_final Y_leaf_final;
    clear V_b_i_all V_full_leaf Omega_leaf_final_n Y_leaf_final_n;
    
    % --------------- Internal levels ---------------
    k_A = k_leaf;
    
    for current_level = num_levels-1:-1:2
        fprintf("CURRENT LEVEL: %d ----------------\n", current_level);
    
        num_blocks = num_blocks / 2;
        block_size = k_A * 2;
    
        if is_unsymmetric
            [Y_current_all, Omega_current_all, Y_current_all_n, Omega_current_all_n] = ...
                get_YOmega_non_leaf(node, current_level, kernel_type, symmetric_kernel_list, 1);
    
            reset_comparison_rng(comparison_seed, current_level, 'U');
            [U_b_i_all, Omega_to_use, Y_to_use, k_A, num_iterations, stats_u] = ...
                building_bases(method, user_tolerance, step_b, mv, n, Y_current_all, Omega_current_all, ...
                    num_blocks, block_size, fixed_pt_leaf, node, current_level, num_levels, 'UV', symmetric_kernel_list, 1, [], collect_diagnostics, sample_distribution, perform_rank_pruning);
            level_stats = append_basis_stat(level_stats, stats_u, 'U');
    
            reset_comparison_rng(comparison_seed, current_level, 'V');
            [V_b_i_all, Omega_to_use_n, Y_to_use_n, k_A_n, num_iterations_n, stats_v] = ...
                building_bases(method, user_tolerance, step_b, mvT, n, Y_current_all_n, Omega_current_all_n, ...
                    num_blocks, block_size, fixed_pt_leaf, node, current_level, num_levels, 'VU', symmetric_kernel_list, 1, [], collect_diagnostics, sample_distribution, perform_rank_pruning);
            level_stats = append_basis_stat(level_stats, stats_v, 'V');
    
            k_A = max(k_A, k_A_n);
            num_iterations = max(num_iterations, num_iterations_n);
    
            % Square/orthonormalize V
            V_full = cell(size(V_b_i_all));
            for idx = 1:length(V_b_i_all)
                V_full{idx} = wy_or_identity(V_b_i_all{idx});
            end
        else
            [Y_current_all, Omega_current_all] = ...
                get_YOmega_non_leaf(node, current_level, kernel_type, symmetric_kernel_list, 1);
    
            reset_comparison_rng(comparison_seed, current_level, 'U');
            [U_b_i_all, Omega_to_use, Y_to_use, k_A, num_iterations, stats_u] = ...
                building_bases(method, user_tolerance, step_b, mv, n, Y_current_all, Omega_current_all, ...
                    num_blocks, block_size, fixed_pt_leaf, node, current_level, num_levels, kernel_type, symmetric_kernel_list, 1, [], collect_diagnostics, sample_distribution, perform_rank_pruning);
            level_stats = append_basis_stat(level_stats, stats_u, 'U');
        end
    
        % Square/orthonormalize U
        U_full = cell(size(U_b_i_all));
        for idx = 1:length(U_b_i_all)
            U_full{idx} = wy_or_identity(U_b_i_all{idx});
        end
    
        % Append per-level table row
        table_rows(end+1, :) = {'', current_level, num_blocks * block_size, num_blocks, block_size, k_A, '', '', '', '', '', '', ''};
    
        % Store current level nodes
        if is_unsymmetric
            node = store_node_results(node, Omega_to_use, Y_to_use, U_full, block_size, current_level, num_levels, k_A, ...
                        Omega_to_use_n, Y_to_use_n, V_full, kernel_type, symmetric_kernel_list, spd_kernel_list, 1);
        else
            node = store_node_results(node, Omega_to_use, Y_to_use, U_full, block_size, current_level, num_levels, k_A, ...
                        [], [], [], kernel_type, symmetric_kernel_list, spd_kernel_list, 1);
        end
    
        clear Y_current_all Omega_current_all Y_current_all_n Omega_current_all_n
        clear U_full V_full U_b_i_all V_b_i_all Omega_to_use Omega_to_use_n Y_to_use Y_to_use_n;
    end
    
    % -------------------- Root --------------------
    nd = 1; left_child = 2*nd; right_child = 2*nd+1;
    node(nd).isLeaf = false;
    
    [Y_i_hat_3_hat_3, Omega_i_hat_3_hat_3] = combine_children_results([node(left_child), node(right_child)], kernel_type, symmetric_kernel_list, []);
    original_indices = union(node(left_child).Indices.skeleton, node(right_child).Indices.skeleton);
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
