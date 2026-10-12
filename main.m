function [node, run_stats] = main(params)
    % main_factor(params)
    % params: struct with fields
    %   .dim            (1 or 2)
    %   .user_tolerance (default: 1e-9)
    %   .step_b         (default: 32)
    %   .num_levels     (default: 11 for 1D, 5 for 2D)
    %   .pt_leaf        (default: 64)
    %   .method         ('naive', 'global', 'iterative', 'householder',
    %                    'safeguarded', or 'corrected'; default: 'householder')
    %   .compute_cond   (default: false; kappa_2 estimate is expensive)
    %   .kernel_type    (optional; otherwise prompt interactively)
    %   .kernel_params  (optional kernel-specific parameters)
    %   .warmup         (default: true)
    %   .collect_diagnostics (default: false)
    %   .comparison_seed (optional; synchronizes per-level sample streams)
    %   .operator_probe_count (default: 3 when diagnostics are collected)
    %   .sample_distribution ('auto', 'real', or 'complex'; default: 'auto')
    %   .rank_pruning   ('all', 'leaf', or 'none'; default: 'all')

    % ---------------- Fill defaults ----------------
    if ~isfield(params,'dim'),            params.dim = 1; end
    if ~isfield(params,'user_tolerance'), params.user_tolerance = 1e-9; end
    if ~isfield(params,'step_b'),         params.step_b = 32; end
    if ~isfield(params,'pt_leaf'),        params.pt_leaf = 64; end
    if ~isfield(params,'num_levels')
        if params.dim==1
            params.num_levels = 11;
        else
            params.num_levels = 5;
        end
    end
    if ~isfield(params,'method'),        params.method = 'householder'; end
    if ~isfield(params,'compute_cond'),  params.compute_cond = false; end
    if ~isfield(params,'kernel_type'),   params.kernel_type = ''; end
    if ~isfield(params,'kernel_params'), params.kernel_params = struct(); end
    if ~isfield(params,'warmup'),        params.warmup = true; end
    if ~isfield(params,'collect_diagnostics'), params.collect_diagnostics = false; end
    if ~isfield(params,'comparison_seed'), params.comparison_seed = []; end
    % The unpreconditioned baseline depends only on A, so the published
    % counts stay valid; recompute it only when asked.
    if ~isfield(params,'unpreconditioned_baseline'), params.unpreconditioned_baseline = false; end
    if ~isfield(params,'operator_probe_count'), params.operator_probe_count = 3; end
    if ~isfield(params,'sample_distribution'), params.sample_distribution = 'auto'; end
    if ~isfield(params,'rank_pruning'), params.rank_pruning = 'all'; end

    % Extract local vars
    dim            = params.dim;
    user_tolerance = params.user_tolerance;
    step_b         = params.step_b;
    num_levels     = params.num_levels;
    pt_leaf        = params.pt_leaf;
    method         = lower(char(params.method));
    compute_cond   = params.compute_cond;
    kernel_type    = lower(char(params.kernel_type));
    kernel_params  = params.kernel_params;
    collect_diagnostics = logical(params.collect_diagnostics);
    comparison_seed = params.comparison_seed;
    operator_probe_count = params.operator_probe_count;
    requested_sample_distribution = lower(char(params.sample_distribution));
    rank_pruning = normalize_rank_pruning(params.rank_pruning);

    valid_methods = ...
        {'naive', 'global', 'iterative', 'householder', 'safeguarded', ...
         'corrected'};
    if ~ismember(method, valid_methods)
        error(['Unknown nullspace method "%s". Choose ''naive'', ' ...
            '''global'', ''iterative'', ''householder'', ' ...
            '''safeguarded'', or ''corrected''.'], method);
    end
    validateattributes(operator_probe_count, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative', 'finite'}, ...
        mfilename, 'params.operator_probe_count');

    fixed_pt_leaf    = pt_leaf;
    WARMUP      = logical(params.warmup);
    symmetric_kernel_list = {'laplace', 'poisson_schur', 'helmholtz_schur', 'matern', 'helmholtz'};
    spd_kernel_list = {'laplace', 'matern', 'poisson_schur'};
    
    switch dim
        case 1
            % ---------- 1D setup ----------
            n = 2^(num_levels - 1) * fixed_pt_leaf;
        
            % 1D builder supports these kernels (see build_test_system.m)
            kernel_list = {'poisson_schur', 'helmholtz_schur', 'matern'};

            % Kernel choice
            if isempty(kernel_type)
                kernel_type = prompt_kernel(kernel_list);
            elseif ~ismember(kernel_type, kernel_list)
                error('Kernel "%s" is unavailable for dim = 1.', kernel_type);
            end
            sample_distribution = resolve_sample_distribution( ...
                requested_sample_distribution, kernel_type);
        
            % Build system (1D builder signature)
            [Y_leaf, b, Omega_leaf, ~, mv, mvT, initial_sketch_time] = ...
                build_test_system(n, fixed_pt_leaf, kernel_type, 1, kernel_params, sample_distribution);
            initial_sketch_count = size(Omega_leaf, 2);
        
            % ---------- Warm-up (optional) ----------
            if WARMUP
                fprintf('Warming up with a small 1D problem...\n');
                
                warmup_levels = 5; 
                warmup_n = 2^(warmup_levels - 1) * fixed_pt_leaf;
                [Y_w, ~, Omega_w, ~, mv_w, mvT_w] = build_test_system( ...
                    warmup_n, fixed_pt_leaf, kernel_type, 1, kernel_params, sample_distribution);
                
                factorize_tree( ...
                    method, warmup_levels, pt_leaf, fixed_pt_leaf, warmup_n, ...
                    kernel_type, symmetric_kernel_list, ...
                    spd_kernel_list, user_tolerance, step_b, ...
                    mv_w, mvT_w, ...
                    Y_w, Omega_w, false, [], sample_distribution, rank_pruning);
            end
        
            % ---------- Factorization ----------
            [node, k_A, table_rows, t_fact, t_mat_vec, n_mat_vec, basis_stats] = factorize_tree( ...
                method, num_levels, pt_leaf, fixed_pt_leaf, n, ...
                kernel_type, symmetric_kernel_list, ...
                spd_kernel_list, user_tolerance, step_b, ...
                mv, mvT, ...
                Y_leaf, Omega_leaf, collect_diagnostics, comparison_seed, ...
                sample_distribution, rank_pruning);
        
        case 2
            % ---------- 2D setup ----------
            leaf_side        = sqrt(fixed_pt_leaf);
            n1d              = 2^(num_levels - 1) * leaf_side;
            n                = n1d^2;
        
            % 2D kernels
            kernel_list = {'laplace', 'poisson_schur', 'helmholtz_schur', 'matern', 'helmholtz'};
        
            % Kernel choice
            if isempty(kernel_type)
                kernel_type = prompt_kernel(kernel_list);
            elseif ~ismember(kernel_type, kernel_list)
                error('Kernel "%s" is unavailable for dim = 2.', kernel_type);
            end
            sample_distribution = resolve_sample_distribution( ...
                requested_sample_distribution, kernel_type);
        
            % Build system
            [Y_leaf, b, Omega_leaf, ~, mv, mvT, initial_sketch_time] = ...
                build_test_system(n1d, pt_leaf, kernel_type, 2, kernel_params, sample_distribution);
            initial_sketch_count = size(Omega_leaf, 2);
        
            % ---------- Warm-up (optional) ----------
            if WARMUP
                fprintf('Warming up with a small 2D problem...\n');
                
                warmup_levels = 3; 
                warmup_n1d = 2^(warmup_levels - 1) * sqrt(fixed_pt_leaf);
                
                [Y_w, ~, Omega_w, ~, mv_w, mvT_w] = build_test_system( ...
                    warmup_n1d, pt_leaf, kernel_type, 2, kernel_params, sample_distribution);
                
                factorize_tree_2D( ...
                    method, warmup_levels, pt_leaf, fixed_pt_leaf, warmup_n1d, ...
                    kernel_type, symmetric_kernel_list, ...
                    spd_kernel_list, user_tolerance, step_b, ...
                    mv_w, mvT_w, ...
                    Y_w, Omega_w, false, [], sample_distribution, rank_pruning);
            end
        
            % ---------- Factorization ----------
            [node, k_A, table_rows, t_fact, t_mat_vec, n_mat_vec, basis_stats] = factorize_tree_2D( ...
                method, num_levels, pt_leaf, fixed_pt_leaf, n1d, ...
                kernel_type, symmetric_kernel_list, ...
                spd_kernel_list, user_tolerance, step_b, ...
                mv, mvT, ...
                Y_leaf, Omega_leaf, collect_diagnostics, comparison_seed, ...
                sample_distribution, rank_pruning);
       
        
        otherwise
            error('dim must be 1 or 2.');
    end

    % ---------------- Condition number of A (optional; expensive) ----------------
    if compute_cond
        [kappa_est, stats] = cond2_from_mv(n, mv, mvT, node, num_levels, kernel_type, dim);
        fprintf('Estimated kappa_2 ≈ %.2e\n', kappa_est);
        disp(stats);
    end
    
    % ---------------- Solve & Krylov ----------------
    [t_solve, residual, solver_name, total_iters, true_relres, ...
        total_iters_np, solver_flag, unpreconditioned_flag] = ...
        solve_and_krylov(node, b, mv, num_levels, kernel_type, spd_kernel_list, symmetric_kernel_list, dim, ...
            params.unpreconditioned_baseline);

    operator_probe_error = NaN;
    hss_probe_time = NaN;
    if collect_diagnostics && operator_probe_count > 0
        probe_seed = 1000003;
        if ~isempty(comparison_seed)
            probe_seed = probe_seed + double(comparison_seed);
        end
        [operator_probe_error, hss_probe_time] = check_matvec( ...
            node, mv, n, num_levels, kernel_type, dim, ...
            operator_probe_count, probe_seed, false);
    end

    % ---------------- Memory stats ----------------
    factor_storage_mb = memory_stats(node);
    
    % ---------------- Table & print ----------------
    table_rows{1, 7}  = n_mat_vec;
    table_rows{1, 8}  = sprintf('%.1e', t_mat_vec);
    table_rows{1, 9}  = sprintf('%.1e', t_fact); 
    table_rows{1, 10}  = sprintf('%.1e', t_solve);
    table_rows{1, 11} = sprintf('%.1e', residual);
    table_rows{1, 12} = sprintf('%.1e', factor_storage_mb);
    table_rows{1, 13} = total_iters;
    table_rows{1, 14} = sprintf('%.1e (%s)', true_relres, solver_name);
    table_rows{1, 15} = total_iters_np;
    
    print_summary_table(table_rows);
    fprintf('Matvec time: initial sketch %.2f s + adaptive updates %.2f s = %.2f s\n', ...
        initial_sketch_time, t_mat_vec, initial_sketch_time+t_mat_vec);
    fprintf("#######################################################################################\n");

    run_stats = basis_stats;
    run_stats.method = method;
    run_stats.sample_distribution = sample_distribution;
    run_stats.kernel_type = kernel_type;
    run_stats.dim = dim;
    run_stats.n = n;
    run_stats.tolerance = user_tolerance;
    run_stats.step_b = step_b;
    run_stats.num_levels = num_levels;
    run_stats.pt_leaf = pt_leaf;
    run_stats.rank_pruning = rank_pruning;
    run_stats.final_rank = k_A;
    % Per-level ranks, so a production run yields the levelwise picture
    % without being re-run. final_rank is only the coarsest level, which
    % summarizes rather than resolves the behaviour across the tree. Stored
    % as a compact string so it survives struct2table and the CSV export.
    run_stats.level_ranks = format_level_ranks(table_rows);
    run_stats.n_matvec = n_mat_vec;
    run_stats.initial_matvec_count = initial_sketch_count;
    run_stats.initial_matvec_time = initial_sketch_time;
    run_stats.adaptive_matvec_time = t_mat_vec;
    run_stats.total_matvec_time = initial_sketch_time + t_mat_vec;
    run_stats.t_matvec = t_mat_vec; % Backward-compatible alias.
    run_stats.t_factor = t_fact;
    run_stats.total_build_time = t_fact + initial_sketch_time + t_mat_vec;
    run_stats.t_solve = t_solve;
    run_stats.solve_residual = residual;
    run_stats.factor_storage_mb = factor_storage_mb;
    run_stats.memory_mb = factor_storage_mb; % Backward-compatible alias.
    run_stats.solver_name = solver_name;
    run_stats.solver_flag = solver_flag;
    run_stats.krylov_iterations = total_iters;
    run_stats.krylov_relres = true_relres;
    run_stats.unpreconditioned_iterations = total_iters_np;
    run_stats.unpreconditioned_solver_flag = unpreconditioned_flag;
    run_stats.operator_probe_count = operator_probe_count;
    run_stats.operator_probe_error = operator_probe_error;
    run_stats.hss_probe_time = hss_probe_time;
end

function scope = normalize_rank_pruning(value)
    if islogical(value) && isscalar(value)
        if value
            scope = 'all';
        else
            scope = 'none';
        end
        return;
    end
    scope = lower(char(value));
    if ~ismember(scope, {'all', 'leaf', 'none'})
        error(['Unknown rank-pruning scope "%s". Choose ''all'', ' ...
            '''leaf'', or ''none''.'], scope);
    end
end

function distribution = resolve_sample_distribution(requested, kernel_type)
    valid = {'auto', 'real', 'complex'};
    if ~ismember(requested, valid)
        error('Unknown sample distribution "%s". Choose ''auto'', ''real'', or ''complex''.', ...
            requested);
    end

    if strcmp(requested, 'auto')
        if strcmp(kernel_type, 'helmholtz')
            distribution = 'complex';
        else
            distribution = 'real';
        end
    else
        distribution = requested;
    end
end

function text = format_level_ranks(table_rows)
%FORMAT_LEVEL_RANKS Compact "level:dim:rank" summary of the per-level rows.
%   factorize_tree appends one row per level as
%   {'', level, total, num_blocks, block_size, k, ...}. Only those four
%   numeric entries are read here, so the summary survives main.m
%   overwriting the trailing columns of the first row.
    text = '';
    if isempty(table_rows) || size(table_rows, 2) < 6
        return;
    end
    parts = strings(0, 1);
    for r = 1:size(table_rows, 1)
        lvl = table_rows{r, 2};
        dim = table_rows{r, 5};
        rnk = table_rows{r, 6};
        if isnumeric(lvl) && isscalar(lvl) && isnumeric(dim) && isscalar(dim) ...
                && isnumeric(rnk) && isscalar(rnk)
            parts(end+1) = sprintf('%d:%d:%d', lvl, dim, rnk); %#ok<AGROW>
        end
    end
    text = char(strjoin(parts, '|'));
end
