function run_block_size_study(output_dir, group, n1d_or_levels, scopes)
%RUN_BLOCK_SIZE_STUDY Rank and cost against the sampling block size.
%   Produces the data behind the block-size tables: selected rank under each
%   pruning scope ('all', 'leaf', 'none') as b varies, together with
%   construction time and the per-level ranks. The 'leaf' scope is the one
%   the paper's theory covers: only there does t:psi-gaussian-iid supply an
%   independent Gaussian look-ahead sample. GROUP is one of 'poisson_1d', 'poisson_2d',
%   'matern', or 'lippmann'; the default runs all four.
%
%   Results are checkpointed after every run, so a job that hits the wall
%   clock can be resubmitted and will resume.

    root = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(output_dir)
        output_dir = fullfile(root, 'comparison_results', 'block_size_study');
    end
    if nargin < 2 || isempty(group), group = 'all'; end
    if ~isfolder(output_dir), mkdir(output_dir); end
    addpath(genpath(root));

    cfgs = block_size_configs();
    if ~strcmpi(group, 'all')
        cfgs = cfgs(strcmpi({cfgs.group}, group));
        if isempty(cfgs)
            error('run_block_size_study:group', 'Unknown group "%s".', group);
        end
    end
    if nargin >= 3 && ~isempty(n1d_or_levels)
        for c = 1:numel(cfgs), cfgs(c).num_levels = n1d_or_levels; end
    end

    block_sizes = [4 8 16 32 64 128];
    % Optional fourth argument: restrict the pruning scopes, so that the
    % slow two-dimensional Poisson sweep can run as three array tasks (one
    % scope each, about 40 minutes instead of two hours) whose raw files
    % merge_block_size_shards joins afterwards.
    if nargin < 4 || isempty(scopes)
        scopes = {'all', 'leaf', 'none'};
    else
        scopes = cellstr(string(scopes));
        bad = setdiff(scopes, {'all', 'leaf', 'none'});
        if ~isempty(bad)
            error('run_block_size_study:scope', 'Unknown pruning scope "%s".', bad{1});
        end
    end
    seeds = [0 1 2 3];   % matched sample streams; tables report the median
    ckpt = fullfile(output_dir, 'block_size_checkpoint.mat');
    code_commit = code_commit_hash();
    raw = struct([]);
    % Resume only a checkpoint this code wrote (see load_checkpoint_if_current).
    companions = fullfile(output_dir, {'block_size_raw.csv', ...
        'block_size_summary.csv', 'block_size_study.mat'});
    [loaded, resumed] = load_checkpoint_if_current(ckpt, companions);
    if resumed
        if isfield(loaded.raw, 'seed')
            raw = loaded.raw;
            fprintf('Resuming with %d completed runs (code %s).\n', numel(raw), code_commit);
        else
            % Pre-seed-loop checkpoint: the row layout changed, so the old
            % runs cannot be appended to. Start over rather than mix them.
            fprintf(['Discarding %d runs from a checkpoint written before ' ...
                'the seed loop was added.\n'], numel(loaded.raw));
        end
    end

    for ic = 1:numel(cfgs)
        cfg = cfgs(ic);
        for b = block_sizes
            for is = 1:numel(scopes)
                scope = scopes{is};
                for seed = seeds
                    if is_done(raw, cfg.name, b, scope, seed)
                        fprintf('skip %s b=%d %s seed=%d\n', cfg.name, b, scope, seed); continue;
                    end
                    params = struct('dim', cfg.dim, 'kernel_type', cfg.kernel, ...
                        'kernel_params', cfg.kernel_params, ...
                        'user_tolerance', cfg.tolerance, 'step_b', b, ...
                        'num_levels', cfg.num_levels, 'pt_leaf', cfg.pt_leaf, ...
                        'method', 'householder', 'rank_pruning', scope, ...
                        'warmup', false, 'compute_cond', false, 'comparison_seed', seed);
                    fprintf('\n=== %s | b = %d | pruning %s | seed %d ===\n', cfg.name, b, scope, seed);
                    t = tic;
                    [~, s] = main(params);
                    row = struct('case_name', string(cfg.name), 'step_b', b, ...
                        'rank_pruning', string(scope), 'seed', seed, 'n', s.n, ...
                        'tolerance', cfg.tolerance, 'pt_leaf', cfg.pt_leaf, ...
                        'num_levels', cfg.num_levels, 'final_rank', s.final_rank, ...
                        'level_ranks', string(s.level_ranks), ...
                        'n_matvec', s.n_matvec, 't_factor', s.t_factor, ...
                        'total_matvec_time', s.total_matvec_time, ...
                        'total_build_time', s.total_build_time, ...
                        'memory_mb', s.memory_mb, 'solve_residual', s.solve_residual, ...
                        'wall_time', toc(t), 'code_commit', string(code_commit));
                    if isempty(raw), raw = row; else, raw(end+1) = row; end %#ok<AGROW>
                    save(ckpt, 'raw', 'code_commit');
                    writetable(struct2table(raw), ...
                        fullfile(output_dir, 'block_size_raw.csv'));
                end
            end
        end
    end
    writetable(struct2table(raw), fullfile(output_dir, 'block_size_raw.csv'));
    write_block_size_summary(raw, output_dir);
    save(fullfile(output_dir, 'block_size_study.mat'), 'raw', 'code_commit');
    fprintf('\nWrote %d runs to %s\n', numel(raw), output_dir);
end

% ------------------------------------------------------------------------
function tf = is_done(raw, name, b, scope, seed)
    tf = false;
    if isempty(raw) || ~isfield(raw, 'seed'), return; end
    tf = any(strcmp(string({raw.case_name}), name) & [raw.step_b] == b & ...
        strcmp(string({raw.rank_pruning}), scope) & [raw.seed] == seed);
end


% ------------------------------------------------------------------------
function cfgs = block_size_configs()
%BLOCK_SIZE_CONFIGS Production-representative sizes, one per family.
%   Leaf 256 throughout: smaller leaves can fall below their own
%   off-diagonal rank, in which case the leaf level is returned at full rank
%   without meeting the tolerance. See the leaf-size section of README.md.
    cfgs = struct( ...
        'name',   {'poisson_1d', 'poisson_2d', 'matern', 'lippmann'}, ...
        'group',  {'poisson_1d', 'poisson_2d', 'matern', 'lippmann'}, ...
        'dim',    {1, 2, 2, 2}, ...
        'kernel', {'poisson_schur', 'poisson_schur', 'matern', 'helmholtz'}, ...
        'kernel_params', {struct(), struct(), struct('nu', 0.5, 'ell', 0.5), struct('potential', 'cavity')}, ...
        'num_levels',  {7, 4, 4, 4}, ...
        'pt_leaf',     {256, 256, 256, 256}, ...
        'tolerance',   {1e-9, 1e-9, 1e-9, 1e-9});
end
