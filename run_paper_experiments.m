function [raw_results, output_dir] = run_paper_experiments( ...
        output_dir, mode, groups, tolerances, n1d_values, method, n_values)
%RUN_PAPER_EXPERIMENTS Reproduce the numerical tables with checkpoints.
%   MODE is 'pilot', 'laptop' (default), 'legacy', 'scaling', or
%   'helmholtz_pilot'. The laptop suite
%   retains every problem family and tolerance in the manuscript while
%   reducing the largest configurations to fit a 16 GB workstation. The
%   legacy suite reproduces the original table dimensions and can require
%   substantially more memory and runtime. The scaling suite contains a
%   small number of larger, moderate-accuracy problems and writes to a
%   separate checkpoint tree.
%
%   GROUPS optionally selects one or more independent problem families:
%   'poisson_1d', 'poisson_2d', 'helmholtz_1d', 'matern_low',
%   'matern_high', or 'lippmann'. It may be a character vector, string array, or cell array
%   of character vectors. The default 'all' runs every family in MODE.
%   TOLERANCES optionally restricts the selected families to one or more
%   positive numeric tolerances. This supports independent checkpoint
%   shards without changing the scientific configurations.
%   N1D_VALUES optionally restricts two-dimensional families to one or more
%   grid widths. Together with TOLERANCES, this supports one-configuration
%   checkpoint shards for long-running two-dimensional experiments.

    if nargin < 2 || isempty(mode)
        mode = 'laptop';
    end
    mode = lower(char(mode));
    if nargin < 3 || isempty(groups)
        groups = "all";
    end
    if nargin < 4
        tolerances = [];
    end
    if nargin < 5
        n1d_values = [];
    end
    % The nullspace method. Section 4 keeps only naive reuse and the
    % Householder row update; 'iterative' is superseded and compresses far
    % worse at tight tolerance, so it is no longer what these tables use.
    if nargin < 6 || isempty(method)
        method = 'householder';
    end
    method = char(method);
    % N_VALUES restricts one-dimensional families by problem size, the way
    % N1D_VALUES restricts two-dimensional ones by grid width. Together they
    % let a launch be staged so the short configurations run and can be
    % inspected before the long ones are committed to a queue.
    if nargin < 7
        n_values = [];
    end

    % Set HSS_XRR_DIAG=1 in the environment to record the conditioning and
    % amplification of every X^rr block alongside the results. Off by
    % default: it costs an SVD per node.
    xrr_diag_requested = strcmp(getenv('HSS_XRR_DIAG'), '1');

    root_dir = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(output_dir)
        output_dir = fullfile(root_dir, 'comparison_results', ...
            ['paper_refresh_' mode]);
    end
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end

    addpath(genpath(root_dir));
    configs = paper_configs(mode);
    [configs, selected_groups] = select_groups(configs, groups);
    [configs, selected_tolerances] = select_tolerances( ...
        configs, tolerances);
    [configs, selected_n1d_values] = select_n1d_values( ...
        configs, n1d_values);
    configs = select_n_values(configs, n_values);
    checkpoint_file = fullfile(output_dir, 'paper_results_checkpoint.mat');
    code_commit = code_commit_hash();
    % Resume only a checkpoint this code wrote (see load_checkpoint_if_current).
    companions = fullfile(output_dir, {'paper_results_raw.csv', ...
        'paper_results_raw_partial.csv'});
    [saved, resumed] = load_checkpoint_if_current(checkpoint_file, companions);
    if resumed
        validate_checkpoint(saved, configs, mode, checkpoint_file);
        rows = upgrade_rows_schema(saved.rows, configs(1), method);
        fprintf('Resuming %d recorded runs from %s (code %s)\n', ...
            numel(rows), checkpoint_file, code_commit);
    else
        rows = struct([]);
    end

    warmed_groups = strings(0, 1);
    for config_id = 1:numel(configs)
        config = configs(config_id);
        if run_is_complete(rows, config)
            fprintf('Skipping completed run %d/%d: %s\n', ...
                config_id, numel(configs), config.name);
            continue;
        end

        warmup = ~any(warmed_groups == string(config.group));
        warmed_groups(end+1, 1) = string(config.group); %#ok<AGROW>
        fprintf('\n=== Run %d/%d: %s ===\n', ...
            config_id, numel(configs), config.name);

        params = struct( ...
            'dim', config.dim, ...
            'kernel_type', config.kernel_type, ...
            'kernel_params', config.kernel_params, ...
            'user_tolerance', config.tolerance, ...
            'step_b', config.step_b, ...
            'num_levels', config.num_levels, ...
            'pt_leaf', config.pt_leaf, ...
            'method', method, ...
            'sample_distribution', 'auto', ...
            'compute_cond', false, ...
            'warmup', warmup, ...
            ... % The probe measures ||(A - A_approx) Xi||_F / ||A Xi||_F, the
            ... % quantity the compression tolerance actually bounds, and is drawn
            ... % after construction so it cannot perturb the sample streams. Its
            ... % cost is excluded from the reported timings.
            'collect_diagnostics', true, ...
            'operator_probe_count', 8, ...
            'comparison_seed', config_seed(config), ...
            'unpreconditioned_baseline', wants_unpreconditioned(config, configs)); %#ok<NASGU>

        if xrr_diag_requested
            global XRR_DIAG %#ok<GVMIS,TLEV>
            XRR_DIAG = struct('enabled', true, 'rows', []);
        end
        rng(config_seed(config), 'twister');
        timer = tic;
        log_text = '';
        try
            log_text = evalc('[node, stats] = main(params);');
            wall_seconds = toc(timer);
            row = success_row(config_id, config, node, stats, wall_seconds, method);
            if xrr_diag_requested
                write_xrr_diagnostic(output_dir, config, method);
            end
            fprintf(['rank %d (max %d) | root %d | matvecs %d | ' ...
                'matvec %.3g s | factor %.3g s | build %.3g s | ' ...
                'solve %.3g s | residual %.3e | storage %.2f MB | ' ...
                '%s iters %g (flag %g) | unpreconditioned %g (flag %g)\n'], ...
                row.final_rank, row.max_rank, row.root_size, row.n_matvec, ...
                row.total_matvec_time, row.factorization_time, ...
                row.total_build_time, row.solve_time, row.solve_residual, ...
                row.factor_storage_mb, char(row.solver_name), ...
                row.krylov_iterations, row.solver_flag, ...
                row.unpreconditioned_iterations, ...
                row.unpreconditioned_solver_flag);
            clear node stats;
        catch caught
            wall_seconds = toc(timer);
            row = failure_row(config_id, config, wall_seconds, caught, method);
            log_text = sprintf('%s\nERROR: %s\n%s', log_text, ...
                caught.message, getReport(caught, 'extended', ...
                'hyperlinks', 'off'));
            warning(caught.identifier, '%s', caught.message);
        end

        log_file = fullfile(output_dir, sprintf('%03d_%s.log', ...
            config_id, config.name));
        write_text(log_file, log_text);
        rows = replace_or_append(rows, row);
        save(checkpoint_file, 'rows', 'configs', 'mode', 'code_commit', ...
            'selected_groups', 'selected_tolerances', ...
            'selected_n1d_values');
        writetable(struct2table(rows), ...
            fullfile(output_dir, 'paper_results_raw_partial.csv'));
        drawnow;
    end

    raw_results = struct2table(rows);
    writetable(raw_results, fullfile(output_dir, 'paper_results_raw.csv'));
    save(fullfile(output_dir, 'paper_results.mat'), ...
        'raw_results', 'configs', 'mode', 'selected_groups', ...
        'selected_tolerances', 'selected_n1d_values');
    fprintf('\nSaved paper-result data to %s\n', output_dir);
    failed = string(raw_results.status) ~= "ok";
    if any(failed)
        error('%d configuration(s) failed; inspect the per-run logs in %s.', ...
            nnz(failed), output_dir);
    end
end

function [configs, selected_n1d_values] = select_n1d_values( ...
        configs, n1d_values)
    available_n1d_values = unique([configs.n1d], 'stable');
    available_n1d_values = available_n1d_values( ...
        isfinite(available_n1d_values));
    if isempty(n1d_values)
        selected_n1d_values = available_n1d_values;
        return;
    end

    validateattributes(n1d_values, {'numeric'}, ...
        {'vector', 'positive', 'finite', 'real', 'integer'}, ...
        mfilename, 'N1D_VALUES');
    selected_n1d_values = unique(n1d_values(:).', 'stable');
    unsupported = setdiff(selected_n1d_values, available_n1d_values);
    if ~isempty(unsupported)
        error(['Requested grid width(s) are unavailable for the ' ...
            'selected group and mode.']);
    end
    configs = configs(ismember([configs.n1d], selected_n1d_values));
    fprintf('Selected grid width(s): %s\n', ...
        strjoin(string(selected_n1d_values), ', '));
end

function configs = select_n_values(configs, n_values)
%SELECT_N_VALUES Restrict one-dimensional configurations by problem size.
%   Two-dimensional configurations carry an explicit n1d and are left alone
%   here; one-dimensional ones record only the tree shape, so the size is
%   recovered as pt_leaf * 2^(num_levels - 1).
    if isempty(n_values)
        return;
    end
    validateattributes(n_values, {'numeric'}, ...
        {'vector', 'positive', 'finite', 'real'}, mfilename, 'N_VALUES');
    wanted = unique(n_values(:).', 'stable');
    is_1d = ~isfinite([configs.n1d]);
    sizes = [configs.pt_leaf] .* 2.^([configs.num_levels] - 1);
    keep = ~is_1d | ismember(sizes, wanted);
    if ~any(keep & is_1d) && any(is_1d)
        error(['Requested problem size(s) are unavailable for the ' ...
            'selected one-dimensional group and mode.']);
    end
    configs = configs(keep);
    fprintf('Selected 1D problem size(s): %s\n', ...
        strjoin(string(wanted), ', '));
end

function [configs, selected_tolerances] = select_tolerances( ...
        configs, tolerances)
    available_tolerances = unique([configs.tolerance], 'stable');
    if isempty(tolerances)
        selected_tolerances = available_tolerances;
        return;
    end

    validateattributes(tolerances, {'numeric'}, ...
        {'vector', 'positive', 'finite', 'real'}, ...
        mfilename, 'TOLERANCES');
    selected_tolerances = unique(tolerances(:).', 'stable');
    unsupported = setdiff(selected_tolerances, available_tolerances);
    if ~isempty(unsupported)
        error('Requested tolerance(s) are unavailable for the selected group.');
    end
    configs = configs(ismember([configs.tolerance], selected_tolerances));
    fprintf('Selected tolerance(s): %s\n', ...
        strjoin(string(selected_tolerances), ', '));
end

function rows = upgrade_rows_schema(rows, example_config, method)
%UPGRADE_ROWS_SCHEMA Add metrics introduced after a checkpoint was written.
%   This keeps existing successful runs reusable after the output schema is
%   extended and prevents dissimilar-structure assignment errors on resume.

    if isempty(rows)
        return;
    end

    template = base_row(0, example_config, method);
    template_fields = fieldnames(template);
    % Rows written before the commit stamp existed must not inherit the
    % current commit from the template; they are marked as unstamped.
    unstamped = ~isfield(rows, 'code_commit');
    for field_id = 1:numel(template_fields)
        name = template_fields{field_id};
        if ~isfield(rows, name)
            default_value = template.(name);
            [rows.(name)] = deal(default_value);
        end
    end
    if unstamped
        [rows.code_commit] = deal("unstamped");
    end

    if all(isnan([rows.factor_storage_mb])) && isfield(rows, 'memory_mb')
        for row_id = 1:numel(rows)
            rows(row_id).factor_storage_mb = rows(row_id).memory_mb;
        end
    end
    for row_id = 1:numel(rows)
        if strlength(string(rows(row_id).solver_name)) == 0 && ...
                string(rows(row_id).status) == "ok"
            if strcmpi(rows(row_id).kernel_type, 'helmholtz')
                rows(row_id).solver_name = "GMRES";
            else
                rows(row_id).solver_name = "PCG";
            end
        end
    end

    rows = orderfields(rows, template);
end

function validate_checkpoint(saved, configs, mode, checkpoint_file)
    if ~isfield(saved, 'configs') || ~isfield(saved, 'mode')
        error(['Checkpoint %s predates configuration validation. Move it ' ...
            'aside or use a new output directory.'], checkpoint_file);
    end
    saved_names = string({saved.configs.name});
    current_names = string({configs.name});
    if ~strcmp(saved.mode, mode) || ~isequal(saved_names, current_names)
        error(['Checkpoint %s belongs to a different mode or group set. ' ...
            'Use a separate output directory.'], checkpoint_file);
    end
end

function [configs, selected_groups] = select_groups(configs, groups)
    if ischar(groups)
        selected_groups = string({groups});
    elseif isstring(groups)
        selected_groups = groups;
    elseif iscellstr(groups)
        selected_groups = string(groups);
    else
        error('GROUPS must be a character vector, string array, or cellstr.');
    end

    selected_groups = lower(strtrim(selected_groups(:)));
    if isempty(selected_groups) || any(strlength(selected_groups) == 0)
        error('GROUPS cannot contain an empty name.');
    end

    available_groups = unique(string({configs.group}), 'stable');
    if any(selected_groups == "all")
        if numel(selected_groups) ~= 1
            error('Use GROUPS="all" by itself.');
        end
        selected_groups = available_groups(:);
    else
        unsupported = setdiff(selected_groups, available_groups, 'stable');
        if ~isempty(unsupported)
            error('Unknown group(s): %s. Available groups: %s.', ...
                strjoin(unsupported, ', '), ...
                strjoin(available_groups, ', '));
        end
        keep = ismember(string({configs.group}), selected_groups);
        configs = configs(keep);
    end

    fprintf('Selected group(s): %s (%d configurations)\n', ...
        strjoin(selected_groups, ', '), numel(configs));
end

function complete = run_is_complete(rows, config)
%RUN_IS_COMPLETE Has CONFIG already produced an ok row in ROWS?
%   Matched by case name, not by position in the current configuration
%   list: two calls that select different single configurations into the
%   same output directory both number their configuration 1, and matching
%   by position made the second call skip its run as "completed" (the
%   512^2, 1e-12 call of a two-configuration job on 2026-09-27).
    complete = false;
    if isempty(rows)
        return;
    end
    select = string({rows.case_name}) == string(config.name) & ...
        string({rows.status}) == "ok";
    complete = any(select);
end

function rows = replace_or_append(rows, row)
    if isempty(rows)
        rows = row;
        return;
    end
    match = find(string({rows.case_name}) == string(row.case_name), 1);
    if isempty(match)
        rows(end+1) = row;
    else
        rows(match) = row;
    end
end

function row = success_row(config_id, config, node, stats, wall_seconds, method)
    row = base_row(config_id, config, method);
    row.status = "ok";
    row.error_message = "";
    row.matlab_version = string(version);
    row.sample_distribution = string(stats.sample_distribution);
    % The probe measures the approximation error the tolerance actually
    % bounds, as against the solve residual, which also carries the
    % elimination's amplification. It is computed whenever
    % collect_diagnostics is on, but used to be dropped on the way to the
    % CSV, so the distinction could not be checked from a production run.
    if isfield(stats, 'operator_probe_error') && ~isempty(stats.operator_probe_error)
        row.operator_probe_error = stats.operator_probe_error;
    end
    row.n = stats.n;
    row.final_rank = stats.final_rank;
    row.max_rank = max([stats.level_stats.rank]);
    row.rank_by_level = serialize_level_stats(stats.level_stats, 'rank');
    row.iterations_by_level = serialize_level_stats( ...
        stats.level_stats, 'iterations');
    row.root_size = size(node(1).X_rr, 1);
    row.n_matvec = stats.n_matvec;
    row.initial_matvec_count = stats.initial_matvec_count;
    row.initial_matvec_time = stats.initial_matvec_time;
    row.adaptive_matvec_time = stats.adaptive_matvec_time;
    row.total_matvec_time = stats.total_matvec_time;
    row.factorization_time = stats.t_factor;
    row.total_build_time = stats.total_build_time;
    row.solve_time = stats.t_solve;
    row.solve_residual = stats.solve_residual;
    row.factor_storage_mb = stats.factor_storage_mb;
    row.memory_mb = stats.memory_mb;
    row.solver_name = string(stats.solver_name);
    row.solver_flag = stats.solver_flag;
    row.krylov_iterations = stats.krylov_iterations;
    row.krylov_relres = stats.krylov_relres;
    row.unpreconditioned_iterations = stats.unpreconditioned_iterations;
    row.unpreconditioned_solver_flag = stats.unpreconditioned_solver_flag;
    row.wall_seconds = wall_seconds;
end

function row = failure_row(config_id, config, wall_seconds, caught, method)
    row = base_row(config_id, config, method);
    row.status = "failed";
    row.error_message = string(caught.message);
    row.matlab_version = string(version);
    row.wall_seconds = wall_seconds;
end

function seed = config_seed(config)
%CONFIG_SEED Sample stream for this configuration, 0 unless one is set.
%   Only families that report medians over streams set it; every other
%   family keeps the single seed-0 run of the protocol in Section 5.1.
    if isfield(config, 'seed') && ~isempty(config.seed)
        seed = config.seed;
    else
        seed = 0;
    end
end

function row = base_row(config_id, config, method)
    % METHOD is passed in explicitly: local functions do not see the main
    % function's variables, and reading it here as a free variable made
    % every row, success or failure, error out after the factorization had
    % already finished (caught by the Stampede3 smoke job of 2026-09-12).
    row = struct( ...
        'config_id', config_id, ...
        'table_id', string(config.table_id), ...
        'case_name', string(config.name), ...
        'status', "", ...
        'error_message', "", ...
        'matlab_version', "", ...
        'hostname', string(getenv('HOSTNAME')), ...
        'slurm_job_id', string(getenv('SLURM_JOB_ID')), ...
        'slurm_array_task_id', string(getenv('SLURM_ARRAY_TASK_ID')), ...
        'omp_num_threads', string(getenv('OMP_NUM_THREADS')), ...
        'method', string(method), ...
        'code_commit', string(code_commit_hash()), ...
        'sample_distribution', "", ...
        'seed', config_seed(config), ...
        'dim', config.dim, ...
        'kernel_type', string(config.kernel_type), ...
        'n', NaN, ...
        'n1d', config.n1d, ...
        'num_levels', config.num_levels, ...
        'pt_leaf', config.pt_leaf, ...
        'step_b', config.step_b, ...
        'tolerance', config.tolerance, ...
        'nu', get_param_or_nan(config.kernel_params, 'nu'), ...
        'ell', get_param_or_nan(config.kernel_params, 'ell'), ...
        'final_rank', NaN, ...
        'max_rank', NaN, ...
        'rank_by_level', "", ...
        'iterations_by_level', "", ...
        'root_size', NaN, ...
        'n_matvec', NaN, ...
        'initial_matvec_count', NaN, ...
        'initial_matvec_time', NaN, ...
        'adaptive_matvec_time', NaN, ...
        'total_matvec_time', NaN, ...
        'factorization_time', NaN, ...
        'total_build_time', NaN, ...
        'solve_time', NaN, ...
        'solve_residual', NaN, ...
        'operator_probe_error', NaN, ...
        'factor_storage_mb', NaN, ...
        'memory_mb', NaN, ...
        'solver_name', "", ...
        'solver_flag', NaN, ...
        'krylov_iterations', NaN, ...
        'krylov_relres', NaN, ...
        'unpreconditioned_iterations', NaN, ...
        'unpreconditioned_solver_flag', NaN, ...
        'wall_seconds', NaN);
end

function write_xrr_diagnostic(output_dir, config, method)
%WRITE_XRR_DIAGNOSTIC Append this run's per-node X^rr rows to a CSV.
    global XRR_DIAG %#ok<GVMIS>
    if isempty(XRR_DIAG) || ~isstruct(XRR_DIAG) || isempty(XRR_DIAG.rows)
        return;
    end
    D = XRR_DIAG.rows;   % [level, size, kappa, smin, smax, norm(X_rs), amp]
    T = table(repmat(string(config.name), size(D, 1), 1), ...
        repmat(string(method), size(D, 1), 1), ...
        D(:,1), D(:,2), D(:,3), D(:,4), D(:,5), D(:,6), D(:,7), ...
        'VariableNames', {'case_name', 'method', 'level', 'block_size', ...
        'kappa', 'sigma_min', 'sigma_max', 'norm_X_rs', 'amplification'});
    f = fullfile(output_dir, 'xrr_diagnostic.csv');
    writetable(T, f, 'WriteMode', ternary(isfile(f), 'append', 'overwrite'), ...
        'WriteVariableNames', ~isfile(f));
end

function out = ternary(c, a, b)
    if c, out = a; else, out = b; end
end

function configs = paper_configs(mode)
    switch mode
        case 'smoke'
            % One small configuration for an end-to-end test of the driver:
            % a five-level 1D Poisson tree at N = 4096, seconds to run.
            configs = make_1d_poisson(2^12, 1e-6);
        case 'pilot'
            configs = [ ...
                make_1d_poisson(2^17, 1e-6), ...
                make_2d_poisson(64, 1e-6), ...
                make_matern(128, 1e-9, 0.5, 0.5, 'matern_low'), ...
                make_lippmann(128, 1e-6)];
        case 'helmholtz_pilot'
            % Exploratory sweep for the 2D Helmholtz Schur complement.
            % Two axes, deliberately small: size at fixed resolution, which
            % raises the frequency and should raise the ranks, and resolution
            % at fixed size, which sweeps from nearly-Poisson (ppw 256) to
            % genuinely oscillatory (ppw 16) at one n. Run this before
            % committing to a table: the method assumes the local blocks
            % X^rr are nonsingular, and an indefinite operator is where that
            % assumption is most likely to fail.
            configs = struct([]);
            for n = 2.^(12:14)
                configs = append_config(configs, ...
                    make_1d_helmholtz(n, 1e-6, 32));
            end
            for ppw = [256 128 64 16]
                configs = append_config(configs, ...
                    make_1d_helmholtz(2^13, 1e-6, ppw));
            end
            for tolerance = [1e-3, 1e-9]
                configs = append_config(configs, ...
                    make_1d_helmholtz(2^13, tolerance, 32));
            end
        case 'laptop'
            configs = struct([]);
            % Schur complement of the 2D Helmholtz equation. Same sizes in
            % both suites: the cap in MAKE_1D_HELMHOLTZ is physical, so the
            % larger machine does not buy a larger problem. Three matched
            % streams, unlike the other application families: here the
            % selected rank and the residual are not well determined by one
            % draw. Three seeds give max ranks of 10, 20 and 16 and
            % residuals from 3.97e-5 to 4.21e-4 at eps = 1e-6, for the same
            % reason the residual moves between machines -- the
            % amplification sits after the compression, so the factorization
            % is far better determined than the residual it delivers.
            for tolerance = [1e-3, 1e-6, 1e-9]
                for n = [4096, 8192, 16384]
                    for seed = 0:2
                        configs = append_config(configs, ...
                            make_1d_helmholtz(n, tolerance, 32, seed));
                    end
                end
            end
            for tolerance = [1e-3, 1e-6, 1e-9, 1e-12]
                for n = 2.^(15:19)
                    configs = append_config(configs, ...
                        make_1d_poisson(n, tolerance));
                end
            end
            for tolerance = [1e-3, 1e-6, 1e-9, 1e-12]
                for n1d = [32, 64, 128]
                    configs = append_config(configs, ...
                        make_2d_poisson(n1d, tolerance));
                end
            end
            for params = [0.5, 0.5; 1.5, 0.1]'
                if params(1) == 0.5
                    group = 'matern_low';
                else
                    group = 'matern_high';
                end
                for tolerance = [1e-9, 1e-12]
                    for n1d = [64, 128, 256]
                        configs = append_config(configs, ...
                            make_matern(n1d, tolerance, ...
                            params(1), params(2), group));
                    end
                end
            end
            for tolerance = [1e-3, 1e-6, 1e-9]
                for n1d = [64, 128, 256]
                    configs = append_config(configs, ...
                        make_lippmann(n1d, tolerance));
                end
            end
            for n1d = [64, 128]
                configs = append_config(configs, ...
                    make_lippmann(n1d, 1e-12));
            end
        case 'legacy'
            configs = struct([]);
            % Schur complement of the 2D Helmholtz equation. Same sizes in
            % both suites: the cap in MAKE_1D_HELMHOLTZ is physical, so the
            % larger machine does not buy a larger problem. Three matched
            % streams, unlike the other application families: here the
            % selected rank and the residual are not well determined by one
            % draw. Three seeds give max ranks of 10, 20 and 16 and
            % residuals from 3.97e-5 to 4.21e-4 at eps = 1e-6, for the same
            % reason the residual moves between machines -- the
            % amplification sits after the compression, so the factorization
            % is far better determined than the residual it delivers.
            for tolerance = [1e-3, 1e-6, 1e-9]
                for n = [4096, 8192, 16384]
                    for seed = 0:2
                        configs = append_config(configs, ...
                            make_1d_helmholtz(n, tolerance, 32, seed));
                    end
                end
            end
            for tolerance = [1e-3, 1e-6, 1e-9, 1e-12]
                for n = 2.^(17:21)
                    configs = append_config(configs, ...
                        make_1d_poisson(n, tolerance));
                end
            end
            % The 512^2 size costs a few hours per tolerance: t_fact grows
            % about tenfold and the matvec count doubles per size step.
            for tolerance = [1e-3, 1e-6, 1e-9, 1e-12]
                for n1d = [64, 128, 256, 512]
                    configs = append_config(configs, ...
                        make_2d_poisson(n1d, tolerance));
                end
            end
            for params = [0.5, 0.5; 1.5, 0.1]'
                if params(1) == 0.5
                    group = 'matern_low';
                else
                    group = 'matern_high';
                end
                for tolerance = [1e-9, 1e-12]
                    for n1d = [128, 256, 512]
                        configs = append_config(configs, ...
                            make_matern(n1d, tolerance, ...
                            params(1), params(2), group));
                    end
                end
            end
            for tolerance = [1e-3, 1e-6]
                for n1d = [128, 256, 512, 1024]
                    configs = append_config(configs, ...
                        make_lippmann(n1d, tolerance));
                end
            end
            % N = 1024^2 at 1e-9 and N = 512^2 at 1e-12 fit one node
            % (about 40 GB and 15 GB of factors).
            for n1d = [128, 256, 512, 1024]
                configs = append_config(configs, ...
                    make_lippmann(n1d, 1e-9));
            end
            for n1d = [128, 256, 512]
                configs = append_config(configs, ...
                    make_lippmann(n1d, 1e-12));
            end
        case 'scaling'
            % Targeted extensions beyond the manuscript's primary suite.
            % Avoid an expensive Cartesian product of sizes and tolerances.
            configs = struct([]);
            for n = [2^22, 2^23]
                configs = append_config(configs, ...
                    make_1d_poisson(n, 1e-9));
            end
            configs = append_config(configs, ...
                make_2d_poisson(512, 1e-6));
            configs = append_config(configs, ...
                make_matern(1024, 1e-9, 0.5, 0.5, 'matern_low'));
            configs = append_config(configs, ...
                make_lippmann(2048, 1e-3));
        otherwise
            error(['Unknown mode "%s". Choose ''smoke'', ''pilot'', ' ...
                '''laptop'', ''legacy'', or ''scaling''.'], ...
                mode);
    end
end

function configs = append_config(configs, config)
    if isempty(configs)
        configs = config;
    else
        configs(end+1) = config; %#ok<AGROW>
    end
end

function config = make_1d_poisson(n, tolerance)
    levels = round(log2(n / 256)) + 1;
    config = make_config('poisson_1d', 'poisson_1d', ...
        sprintf('poisson_1d_N%d_tol_%s', n, tolerance_tag(tolerance)), ...
        1, 'poisson_schur', struct(), levels, 256, 16, tolerance, NaN);
end

function config = make_2d_poisson(n1d, tolerance)
    levels = round(log2(n1d / 16)) + 1;
    config = make_config('poisson_2d', 'poisson_2d', ...
        sprintf('poisson_2d_N%d_tol_%s', n1d^2, tolerance_tag(tolerance)), ...
        2, 'poisson_schur', struct(), levels, 256, 32, tolerance, n1d);
end

function config = make_matern(n1d, tolerance, nu, ell, group)
    levels = round(log2(n1d / 16)) + 1;
    name = sprintf('%s_N%d_tol_%s', group, n1d^2, ...
        tolerance_tag(tolerance));
    config = make_config(group, group, name, 2, 'matern', ...
        struct('nu', nu, 'ell', ell), levels, 256, 32, tolerance, n1d);
end

function config = make_1d_helmholtz(n, tolerance, ppw, seed)
%MAKE_1D_HELMHOLTZ Schur complement of the 2D Helmholtz equation.
%   Same geometry as MAKE_1D_POISSON -- a 51-by-n strip split 25 | 1 | 25,
%   leaving an n-point separator -- but with the Helmholtz shift, so the
%   half-domains and the Schur complement are symmetric indefinite rather
%   than positive definite.
%
%   PPW is the number of grid points per wavelength, held fixed as n grows,
%   so the domain spans n/PPW wavelengths and the off-diagonal ranks grow
%   with frequency. PPW = 32 matches the 3D Helmholtz box and the
%   Lippmann-Schwinger problem. Raising PPW makes the problem easier, not
%   larger: at PPW near 2^15 the shift is negligible and this reduces to the
%   Poisson case.
    if nargin < 3 || isempty(ppw), ppw = 32; end
    if nargin < 4 || isempty(seed), seed = 0; end
    % The size is capped, and the cap is physical rather than a budget.
    % Dirichlet walls on all four sides make the strip a closed resonator,
    % and its 51*n resonances are packed into a fixed interval, so the mean
    % gap between them is about 8/(51*n): 3.8e-5 at n = 4096 but 7.5e-8 at
    % n = 2^21. Below the cap a wavenumber can sit clear of every resonance;
    % above it, no wavenumber can, and the half-domain solves are then
    % near-singular no matter how k is chosen (verified by scanning
    % points-per-wavelength over [30, 34] at n = 32768: the best attainable
    % distance to a resonance was 1.9e-6, against 1.66e-6 at ppw = 32, and
    % the delivered residual was 781 times the tolerance). Removing the cap
    % needs absorption -- a complex shift -- which makes the operator
    % complex symmetric and is a different experiment.
    max_n = 16384;
    if n > max_n
        error('run_paper_experiments:helmholtzSize', ...
            ['helmholtz_1d is capped at n = %d; %d was requested. Above ' ...
             'the cap the closed-strip resonances are denser than any ' ...
             'choice of wavenumber can avoid.'], max_n, n);
    end
    levels = round(log2(n / 256)) + 1;
    config = make_config('helmholtz_1d', 'helmholtz_1d', ...
        sprintf('helmholtz_1d_N%d_ppw%d_tol_%s_s%d', n, ppw, ...
            tolerance_tag(tolerance), seed), ...
        1, 'helmholtz_schur', struct('points_per_wavelength', ppw), ...
        levels, 256, 16, tolerance, NaN);
    config.seed = seed;
end

function tf = wants_unpreconditioned(config, configs)
%WANTS_UNPRECONDITIONED Run the unpreconditioned Krylov baseline for CONFIG?
%   The count depends on N alone, not on the tolerance or the seed, and for
%   the sizes a laptop can hold it is measured locally
%   (run_cavity_unpreconditioned.m, run_helmholtz_strip_unpreconditioned.m,
%   run_matern_unpreconditioned.m). Only the Lippmann-Schwinger group needs
%   Stampede3's memory for its largest size (the unrestarted Krylov basis
%   at N = 1024^2), so the baseline runs there once: on the first selected
%   configuration at the group's largest n1d. Every other row is NaN in the
%   CSV and the table takes the column from the local scripts.
    tf = false;
    if ~strcmpi(config.group, 'lippmann'), return; end
    % HSS_SKIP_UNPRECOND=1 skips it everywhere: the counts are already
    % measured (23, 58, 214, 785 at N = 128^2 ... 1024^2) and do not depend
    % on the code version, so a rerun for new timings need not repeat the
    % hours-long N = 1024^2 baseline. A sharded rerun would otherwise run
    % the baseline once per shard, at each shard's largest size.
    if strcmp(getenv('HSS_SKIP_UNPRECOND'), '1'), return; end
    same = configs(strcmpi({configs.group}, config.group));
    n_max = max([same.n1d]);
    if config.n1d ~= n_max, return; end
    first = same(find([same.n1d] == n_max, 1, 'first'));
    tf = strcmp(first.name, config.name);
end

function config = make_lippmann(n1d, tolerance)
    % The potential is the near-resonant cavity of Gopal and Martinsson
    % (arXiv:2007.12718, Sec. 6.2), see build_test_system.m and
    % run_cavity_unpreconditioned.m; struct('potential', 'gaussian_bump')
    % selects the alternative Gaussian-bump potential.
    levels = round(log2(n1d / 16)) + 1;
    config = make_config('lippmann', 'lippmann', ...
        sprintf('lippmann_N%d_tol_%s', n1d^2, tolerance_tag(tolerance)), ...
        2, 'helmholtz', struct('potential', 'cavity'), levels, 256, 32, tolerance, n1d);
end

function config = make_config(table_id, group, name, dim, kernel_type, ...
        kernel_params, num_levels, pt_leaf, step_b, tolerance, n1d)
    config = struct( ...
        'table_id', table_id, ...
        'group', group, ...
        'name', name, ...
        'dim', dim, ...
        'kernel_type', kernel_type, ...
        'kernel_params', kernel_params, ...
        'num_levels', num_levels, ...
        'pt_leaf', pt_leaf, ...
        'step_b', step_b, ...
        'tolerance', tolerance, ...
        'n1d', n1d, ...
        ... % Sample stream. Zero for every family except the ones that
        ... % report medians over matched streams, which override it.
        'seed', 0);
end

function tag = tolerance_tag(tolerance)
    tag = strrep(sprintf('%.0e', tolerance), '-', 'm');
end

function value = get_param_or_nan(params, name)
    if isfield(params, name)
        value = params.(name);
    else
        value = NaN;
    end
end

function value = serialize_level_stats(level_stats, field)
    if isempty(level_stats)
        value = "";
        return;
    end
    tokens = strings(1, numel(level_stats));
    for idx = 1:numel(level_stats)
        tokens(idx) = sprintf('%s:L%d:%g', level_stats(idx).basis, ...
            level_stats(idx).level, level_stats(idx).(field));
    end
    value = strjoin(tokens, ';');
end

function write_text(path, contents)
    file_id = fopen(path, 'w');
    if file_id < 0
        error('Could not open log file %s for writing.', path);
    end
    cleanup = onCleanup(@() fclose(file_id));
    fwrite(file_id, contents, 'char');
end
