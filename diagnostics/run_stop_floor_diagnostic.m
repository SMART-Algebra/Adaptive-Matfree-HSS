function results = run_stop_floor_diagnostic(which)
%RUN_STOP_FLOOR_DIAGNOSTIC Effect of the level-mean floor on the stopping test.
%   RESULTS = RUN_STOP_FLOOR_DIAGNOSTIC('cavity') runs the cavity-potential
%   Lippmann-Schwinger problem (run_cavity_unpreconditioned.m) with the
%   floor on, plus the Gaussian bump with it off and on.
%   RESULTS = RUN_STOP_FLOOR_DIAGNOSTIC('regression') runs the paper's real
%   families with the floor off and on, same seed, to check that it changes
%   none of the published tables.
%
%   The floor (on by default in building_bases.m; the global
%   STOP_FLOOR_MODE = 'none' switches it off)
%   judges a node against max(its own sampled energy, the level's mean
%   sampled energy). Without it, a block whose norm sits at matvec roundoff
%   -- a leaf where a compactly supported potential vanishes -- can never
%   pass the purely relative test rho_i <= eps/sqrt(b) ||G_i||: its
%   threshold is ~1e-17 to 1e-23 while its residual is ~1e-14 of FFT noise
%   that shrinks only as each batch consumes b of the leaf's noise
%   dimensions. It reaches rank 256 = full, and because the level width is
%   common to all nodes, the whole level is stored dense. At N=256^2, 30
%   (eps=1e-3) to 64 (1e-9) of the 256 leaves failed this way, which is what
%   made the cavity runs take an hour and, at eps=1e-9, exhaust memory.
%
%   Results, local single run, seed 0, 2026-09-26:
%
%   Cavity N=128^2  (before -> after the floor)
%     eps=1e-3: leaf rank 256 -> 51,  t_fact 52 s -> 6.0 s,  relres 4.0e-4 -> 3.8e-4, n_it 3 -> 3
%     eps=1e-6: leaf rank 256 -> 106, t_fact 64 s -> 15.4 s, relres 1.5e-7 -> 2.5e-7, n_it 2 -> 2
%     eps=1e-9: leaf rank 256 -> 119, t_fact 67 s -> 23.9 s, relres 7.6e-11 -> 2.6e-10, n_it 1 -> 2
%   Cavity N=256^2, eps=1e-6: t_fact 3994 s -> 110 s, level ranks 111/191/256/192
%     (root 768 against the Gaussian bump's 1280 in Table 7), relres 5.6e-7 ->
%     1.8e-6, mem 1071 MB, n_it 2.
%   Gaussian bump N=128^2, eps=1e-6: ranks 112/224/224, n_mv 1120, mem 277 MB
%     identical with and without; relres 1.6e-6 -> 6.7e-7 (run-to-run spread).
%
%   Regression, floor off -> on (level ranks | n_mv | relres):
%     1D Poisson N=8192, 1e-6 and 1e-9: identical
%     1D Helmholtz strip N=4096, 1e-6: 14/14/14/14 -> 14/14/14/13, relres 5.5e-5 -> 5.9e-5
%     2D Poisson N=128^2, 1e-6: identical (177/384/416, n_mv 1952, relres 3.1e-5)
%     2D Matern (0.5,0.5) N=128^2, 1e-9: 175/377/384 -> 174/378/384, n_mv 1892 -> 1896,
%       relres 1.2e-10 -> 1.05e-10
%     2D Matern (1.5,0.1) N=128^2, 1e-9: identical (182/414/448, n_mv 2104, relres 1.75e-9)
%     2D Poisson N=128^2, 1e-12 (dense leaves): identical (256/606/704, n_mv 3128)
%     2D Matern (1.5,0.1) N=128^2, 1e-12: identical (229/541/608, n_mv 2772)
%     2D Poisson N=256^2, 1e-9: identical (218/512/1056/1088, n_mv 5312), relres 9.60e-8 -> 9.70e-8
%     2D Matern (1.5,0.1) N=256^2, 1e-9: identical (161/352/640/736, n_mv 3296, the published
%       Table 6 row), relres 2.805e-8 -> 2.815e-8
%   The floor binds only on nodes whose sampled energy is below the level
%   mean; the level width is the maximum over nodes, so the published ranks
%   are unaffected up to the +-1 seen on the easy Matern problem.

    if nargin < 1 || isempty(which), which = 'cavity'; end
    addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));  % repository root
    global STOP_FLOOR_MODE %#ok<GVMIS>
    saved_mode = STOP_FLOOR_MODE;
    cleanup = onCleanup(@() restore_mode(saved_mode)); %#ok<NASGU>

    switch lower(which)
        case 'cavity'
            % name, potential, n1d, tol, mode
            cases = { ...
                'bump_128_1e-6_none',  'gaussian_bump', 128, 1e-6, 'none'; ...
                'bump_128_1e-6_floor', 'gaussian_bump', 128, 1e-6, ''; ...
                'cav_128_1e-3_floor',  'cavity', 128, 1e-3, ''; ...
                'cav_128_1e-6_floor',  'cavity', 128, 1e-6, ''; ...
                'cav_128_1e-9_floor',  'cavity', 128, 1e-9, ''; ...
                'cav_256_1e-6_floor',  'cavity', 256, 1e-6, ''};
            cfgs = cell(size(cases, 1), 1);
            for c = 1:size(cases, 1)
                cfgs{c} = struct('name', cases{c,1}, 'mode', cases{c,5}, ...
                    'params', lippmann_params(cases{c,2}, cases{c,3}, cases{c,4}));
            end
        case 'regression'
            % name, dim, kernel, kernel_params, n1d, n, step_b, tol
            base = { ...
                'poisson1d_8192_1e-6',  1, 'poisson_schur',   struct(), NaN,  8192, 16, 1e-6; ...
                'poisson1d_8192_1e-9',  1, 'poisson_schur',   struct(), NaN,  8192, 16, 1e-9; ...
                'helm1d_4096_1e-6',     1, 'helmholtz_schur', struct('points_per_wavelength', 32), NaN, 4096, 16, 1e-6; ...
                'poisson2d_128_1e-6',   2, 'poisson_schur',   struct(), 128, NaN, 32, 1e-6; ...
                'matern_easy_128_1e-9', 2, 'matern', struct('nu', 0.5, 'ell', 0.5), 128, NaN, 32, 1e-9; ...
                'matern_hard_128_1e-9', 2, 'matern', struct('nu', 1.5, 'ell', 0.1), 128, NaN, 32, 1e-9};
            modes = {'', ''};
            cfgs = {};
            for c = 1:size(base, 1)
                dim = base{c,2};
                if dim == 1
                    levels = round(log2(base{c,6} / 256)) + 1;
                else
                    levels = round(log2(base{c,5} / 16)) + 1;
                end
                params = struct('dim', dim, 'kernel_type', base{c,3}, ...
                    'kernel_params', base{c,4}, 'user_tolerance', base{c,8}, ...
                    'step_b', base{c,7}, 'num_levels', levels, 'pt_leaf', 256, ...
                    'method', 'householder', 'sample_distribution', 'auto', ...
                    'compute_cond', false, 'warmup', false, ...
                    'collect_diagnostics', false, 'operator_probe_count', 0, ...
                    'comparison_seed', 0, 'unpreconditioned_baseline', false);
                for m = 1:2
                    cfgs{end+1} = struct('name', sprintf('%s floor=%s', base{c,1}, modes{m}), ...
                        'mode', modes{m}, 'params', params); %#ok<AGROW>
                end
            end
        otherwise
            error('run_stop_floor_diagnostic:which', ...
                'WHICH must be ''cavity'' or ''regression'', got ''%s''.', which);
    end

    results = struct('name', {}, 'level_ranks', {}, 'n_matvec', {}, 't_fact', {}, ...
        'solve_residual', {}, 'factor_storage_mb', {}, 'krylov_iterations', {}, 'wall_seconds', {});
    for c = 1:numel(cfgs)
        STOP_FLOOR_MODE = cfgs{c}.mode;
        rng(0, 'twister');
        timer = tic;
        fprintf('\n=== %s ===\n', cfgs{c}.name);
        try
            [~, stats] = main(cfgs{c}.params);
            row = numel(results) + 1;
            results(row).name = cfgs{c}.name;
            results(row).level_ranks = stats.level_ranks;
            results(row).n_matvec = stats.n_matvec;
            results(row).t_fact = stats.t_factor;
            results(row).solve_residual = stats.solve_residual;
            results(row).factor_storage_mb = stats.factor_storage_mb;
            results(row).krylov_iterations = stats.krylov_iterations;
            results(row).wall_seconds = toc(timer);
            fprintf(['RESULT %s: level_ranks=%s n_mv=%d t_fact=%.1f relres=%.3e ' ...
                'mem=%.1fMB n_it=%d wall=%.1fs\n'], cfgs{c}.name, stats.level_ranks, ...
                stats.n_matvec, stats.t_factor, stats.solve_residual, ...
                stats.factor_storage_mb, stats.krylov_iterations, toc(timer));
        catch caught
            fprintf('RESULT %s: FAILED after %.1fs: %s\n', cfgs{c}.name, toc(timer), caught.message);
        end
    end
end

function params = lippmann_params(potential, n1d, tolerance)
    levels = round(log2(n1d / 16)) + 1;
    params = struct('dim', 2, 'kernel_type', 'helmholtz', ...
        'kernel_params', struct('potential', potential), ...
        'user_tolerance', tolerance, 'step_b', 32, 'num_levels', levels, ...
        'pt_leaf', 256, 'method', 'householder', 'sample_distribution', 'auto', ...
        'compute_cond', false, 'warmup', false, 'collect_diagnostics', false, ...
        'operator_probe_count', 0, 'comparison_seed', 0, ...
        'unpreconditioned_baseline', false);
end

function restore_mode(saved_mode)
    global STOP_FLOOR_MODE %#ok<GVMIS>
    STOP_FLOOR_MODE = saved_mode;
end
