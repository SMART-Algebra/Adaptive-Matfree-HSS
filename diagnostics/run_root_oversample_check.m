function results = run_root_oversample_check(cases, factors)
%RUN_ROOT_OVERSAMPLE_CHECK Does oversampling the root recovery change accuracy?
%   The 2D driver draws samples after each level until they cover the
%   parent dimension, 4 k_A in the quadtree. At the root nothing more is
%   drawn, so when the rank search below the root stops with fewer samples
%   than 4 k_A the root block A^(0) is recovered from a SQUARE system
%   (Table 7 cavity rows at eps = 1e-12: n_mv = |A^(0)| = 4608 and 9984).
%   Several other published rows are nearly square: 3D Poisson N = 64^2 has
%   n_mv/|A^(0)| = 1.027, 1.014, 1.045 at eps = 1e-6, 1e-9, 1e-12, and the
%   Matern (1.5, 0.1) row at N = 512^2, eps = 1e-9, has 1.039.
%
%   This script rebuilds the given CASES (struct array with fields kernel,
%   kernel_params, n1d, tol; default: the three 3D Poisson 64^2 rows and the
%   cavity at 128^2, eps = 1e-12) with the global HSS_ROOT_OVERSAMPLE set to
%   each of FACTORS (default [1 1.25]; the driver's default since
%   2026-10-10 is 1.1, and 1 restores the earlier behavior) and reports n_mv, the root size, the
%   direct-solve residual, the operator probe error and the preconditioned
%   Krylov count, so the square-root effect can be read off directly.
%
%   Local, single configuration per call of MAIN, seed 0 throughout.

    here = fileparts(fileparts(mfilename('fullpath')));  % repository root
    addpath(genpath(here));
    global HSS_ROOT_OVERSAMPLE %#ok<GVMIS>
    if nargin < 1 || isempty(cases)
        cases = [ ...
            struct('kernel', 'poisson_schur', 'kernel_params', struct(), 'n1d', 64, 'tol', 1e-6), ...
            struct('kernel', 'poisson_schur', 'kernel_params', struct(), 'n1d', 64, 'tol', 1e-9), ...
            struct('kernel', 'poisson_schur', 'kernel_params', struct(), 'n1d', 64, 'tol', 1e-12), ...
            struct('kernel', 'helmholtz', 'kernel_params', struct('potential', 'cavity'), 'n1d', 128, 'tol', 1e-12)];
    end
    if nargin < 2 || isempty(factors), factors = [1, 1.25]; end

    results = [];
    for c = 1:numel(cases)
        cs = cases(c);
        levels = round(log2(cs.n1d / 16)) + 1;
        for f = factors
            HSS_ROOT_OVERSAMPLE = f;
            params = struct('dim', 2, 'kernel_type', cs.kernel, ...
                'kernel_params', cs.kernel_params, ...
                'user_tolerance', cs.tol, 'step_b', 32, ...
                'num_levels', levels, 'pt_leaf', 256, ...
                'method', 'householder', 'sample_distribution', 'auto', ...
                'compute_cond', false, 'warmup', false, ...
                'collect_diagnostics', false, 'operator_probe_count', 8, ...
                'comparison_seed', 0, 'unpreconditioned_baseline', false);
            rng(0, 'twister');
            fprintf('\n=== %s n1d=%d eps=%g root oversample %.2f ===\n', cs.kernel, cs.n1d, cs.tol, f);
            [node, stats] = main(params);
            root = size(node(1).X_rr, 1);
            r = struct('kernel', cs.kernel, 'n1d', cs.n1d, 'tol', cs.tol, 'factor', f, ...
                'root', root, 'n_mv', stats.n_matvec, 'ratio', stats.n_matvec / root, ...
                'relres', stats.solve_residual, 'probe', stats.operator_probe_error, ...
                'n_it', stats.krylov_iterations, 't_fact', stats.t_factor, ...
                'mem_mb', stats.factor_storage_mb);
            if isempty(results), results = r; else, results(end+1) = r; end %#ok<AGROW>
            fprintf('RESULT %-14s n1d=%4d eps=%-6g f=%.2f root=%5d n_mv=%5d ratio=%.3f relres=%.3e probe=%.3e n_it=%d t_fact=%.1f\n', ...
                cs.kernel, cs.n1d, cs.tol, f, root, stats.n_matvec, stats.n_matvec / root, ...
                stats.solve_residual, stats.operator_probe_error, stats.krylov_iterations, stats.t_factor);
        end
    end
    HSS_ROOT_OVERSAMPLE = [];  % back to the driver default
end
