function results = run_helmholtz_strip_unpreconditioned()
%RUN_HELMHOLTZ_STRIP_UNPRECONDITIONED Unpreconditioned GMRES baseline for
%   the 2D Helmholtz Schur complement (Table 5, Section 5.3), run locally.
%
%   This can run on a workstation because it reports only an iteration
%   count, not a time: solve_and_krylov.m's unpreconditioned branch acts on
%   the bare operator and right-hand side alone (no HSS factorization, no
%   preconditioner Mfun), so the count is a property of the discretized
%   operator and is machine-independent, unlike every timing column in the
%   manuscript's tables. It is also independent of the compression
%   tolerance and the sample seed, since neither affects the operator or
%   the right-hand side b = mv(original_x); Table 7's caption already
%   states the analogous fact for Lippmann-Schwinger ("depends on N only").
%   So only 3 runs are needed here, not the 27 configurations
%   (3 sizes x 3 tolerances x 3 seeds) the production family covers.
%
%   Settings: no restart, tol=1e-12, maxit=10000 (a cap of 1000 is not
%   enough at these sizes).
%
%   Result (single core): N=4096: 1370 iters, N=8192: 2604 iters,
%   N=16384: 5100 iters, all to relres about 9.5e-13. MATLAB clamps
%   maxit to N for unrestarted GMRES (a Krylov space cannot exceed the
%   system size); this only bound the N=4096 and N=8192 runs (clamped to
%   4096 and 8192 respectively) and did not bind the actual convergence
%   point in either case.

    addpath(genpath(fileparts(mfilename('fullpath'))));
    ppw = 32;
    sizes = [4096, 8192, 16384];
    tol = 1e-12;
    maxit = 10000;

    results = struct('n', {}, 'iters', {}, 'flag', {}, 'relres', {});
    for idx = 1:numel(sizes)
        n = sizes(idx);
        [~, b, ~, ~, mv, ~] = build_test_system(n, 256, 'helmholtz_schur', 1, ...
            struct('points_per_wavelength', ppw), 'real');
        [~, flag, relres, iter, resvec] = gmres(mv, b, [], tol, maxit);
        fprintf('N=%5d: flag=%d iters=%d relres=%.3e resvec(end)/||b||=%.3e\n', ...
            n, flag, iter(2), relres, resvec(end)/norm(b));
        results(idx).n = n; %#ok<AGROW>
        results(idx).iters = iter(2); %#ok<AGROW>
        results(idx).flag = flag; %#ok<AGROW>
        results(idx).relres = relres; %#ok<AGROW>
    end
end
