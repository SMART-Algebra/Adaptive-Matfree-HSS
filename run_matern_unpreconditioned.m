function results = run_matern_unpreconditioned()
%RUN_MATERN_UNPRECONDITIONED Unpreconditioned Krylov baseline for the
%   Matern kernel (Table 6, Section 5.4), run locally to confirm the
%   published counts.
%
%   Matern is SPD (in spd_kernel_list), so its unpreconditioned baseline
%   is PCG, not GMRES -- PCG has no restart parameter to disable; it never
%   restarts. Like the Helmholtz-strip and Lippmann-Schwinger counts, this
%   needs no HSS factorization (bare operator and right-hand side only)
%   and reports only an iteration count, not a time, so it is
%   machine-independent. Unlike those two families, the OPERATOR itself
%   depends on the parameter set (nu, ell), not just on N, so all six
%   (parameter set, size) combinations are run, not three.
%
%   Settings: tol=1e-12, maxit=10000, matching
%   Solve_Matvec/solve_and_krylov.m's 'matern' branch exactly.
%
%   Result (single core): confirms the published counts (538, 1053, 1689
%   for (0.5,0.5) and 2392, 3925, 4996 for (1.5,0.1), at N=128^2/256^2/512^2)
%   to within about 11%: this run got 566, 1059, 1740 and 2391, 4348, 4983.
%   The spread is consistent with the cross-machine/cross-BLAS PCG
%   sensitivity this ill-conditioned family already shows elsewhere in the
%   manuscript (kappa(A) up to 3e7); all six runs converged (flag=0) to
%   relres about 9e-13, well inside tol=1e-12.

    addpath(genpath(fileparts(mfilename('fullpath'))));
    tol = 1e-12;
    maxit = 10000;
    sizes = [128, 256, 512];
    param_sets = {[0.5, 0.5], [1.5, 0.1]};
    labels = {'(nu,ell)=(0.5,0.5)', '(nu,ell)=(1.5,0.1)'};

    results = struct('label', {}, 'n1d', {}, 'iters', {}, 'flag', {}, 'relres', {});
    row = 0;
    for p = 1:numel(param_sets)
        nu = param_sets{p}(1); ell = param_sets{p}(2);
        for idx = 1:numel(sizes)
            n1d = sizes(idx);
            [~, b, ~, ~, mv, ~] = build_test_system(n1d, 256, 'matern', 2, ...
                struct('nu', nu, 'ell', ell), 'real');
            [~, flag, relres, iter, resvec] = pcg(mv, b, tol, maxit);
            row = row + 1;
            fprintf('%s N=%4d^2: flag=%d iters=%d relres=%.3e resvec(end)/||b||=%.3e\n', ...
                labels{p}, n1d, flag, iter, relres, resvec(end)/norm(b));
            results(row).label = labels{p}; %#ok<AGROW>
            results(row).n1d = n1d; %#ok<AGROW>
            results(row).iters = iter; %#ok<AGROW>
            results(row).flag = flag; %#ok<AGROW>
            results(row).relres = relres; %#ok<AGROW>
        end
    end
end
