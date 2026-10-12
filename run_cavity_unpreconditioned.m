function results = run_cavity_unpreconditioned(sizes, mem_budget_bytes)
%RUN_CAVITY_UNPRECONDITIONED Unpreconditioned GMRES on the cavity-potential
%   Lippmann-Schwinger problem of Table 7 (Gopal & Martinsson 2022,
%   arXiv:2007.12718, Section 6.2). That paper reports unpreconditioned
%   GMRES as "unfeasibly high" for this problem, due to near-resonant
%   behavior, but gives no counts; this measures them for our
%   discretization.
%
%   Discretization and right-hand side as in build_test_system.m
%   (kind='helmholtz'): a uniform n1d-by-n1d grid, piecewise-constant
%   collocation, ppw=32 grid points per wavelength, incident field
%   u0(x)=exp(i 2 pi q x1) with q=n1d/32 wavelengths per unit length,
%   Green's function via a 2D FFT (block Toeplitz). The potential is
%   b(r,theta)=(1-sin(0.5 theta)^500)*exp(-2000*(0.1-r^2)^2), a near-total
%   reflector ring at r^2=0.1 with a thin slot; the Gaussian bump
%   exp(-32||x-c||^2) is the alternative potential.
%
%   The paper's own example uses q=8 wavelengths across the domain
%   (kappa=50.27=2*pi*8.0), matching q=8 here (n1d=256). We also sweep
%   n1d=[128,256,512,1024] (q=4,8,16,32), the sizes of Table 7.
%
%   Settings: no restart, tol=1e-12, maxit=10000, single core, with the
%   memory-aware cap on maxit described below (unrestarted GMRES stores
%   one N-vector per iteration).
%
%   Result (2026-09-26, single core; the counts for the earlier Gaussian-bump
%   potential at the same size in parentheses):
%     N= 128^2 (q= 4):   23 iterations  (17)
%     N= 256^2 (q= 8):   58             (28)
%     N= 512^2 (q=16):  211             (55)   maxit capped at 2861, did not bind;
%                      214 on Stampede3 (2026-09-27):
%                      the count is machine-independent up to roundoff
%     N=1024^2 (q=32): >715, flag 1, relres 1.9e-8 at the memory cap (142);
%                      converged at 785 (flag 0) on Stampede3 under an
%                      80 GB budget (Stampede3, 2026-09-27)
%   The gap widens with the frequency, 1.4x, 2.1x, 3.8x, 5.5x.

    addpath(genpath(fileparts(mfilename('fullpath'))));
    tol = 1e-12;
    % SIZES and MEM_BUDGET_BYTES are optional: the defaults are the laptop
    % sweep; hpc/stampede3/lippmann_unprecond_1024_icx.slurm calls
    % run_cavity_unpreconditioned(1024, 80e9) for the one count the laptop
    % cannot converge (the Table 7 production job skips this baseline).
    if nargin < 1 || isempty(sizes), sizes = [128, 256, 512, 1024]; end
    if nargin < 2 || isempty(mem_budget_bytes), mem_budget_bytes = 12e9; end

    results = struct('n1d', {}, 'iters', {}, 'flag', {}, 'relres', {}, ...
        'maxit_used', {}, 'maxit_capped_by_memory', {});
    for idx = 1:numel(sizes)
        n1d = sizes(idx);
        N = n1d^2;
        maxit_mem_limit = floor(mem_budget_bytes / (N * 16));
        maxit = min(10000, maxit_mem_limit);
        capped = maxit < 10000;
        [~, b, ~, ~, mv, ~] = build_test_system(n1d, 256, 'helmholtz', 2, ...
            struct('potential', 'cavity'), 'complex');
        [~, flag, relres, iter, resvec] = gmres(mv, b, [], tol, maxit);
        fprintf('N=%4d^2 (q=%d): maxit=%d flag=%d iters=%d relres=%.3e resvec(end)/||b||=%.3e\n', ...
            n1d, n1d/32, maxit, flag, iter(2), relres, resvec(end)/norm(b));
        results(idx).n1d = n1d; %#ok<AGROW>
        results(idx).iters = iter(2); %#ok<AGROW>
        results(idx).flag = flag; %#ok<AGROW>
        results(idx).relres = relres; %#ok<AGROW>
        results(idx).maxit_used = maxit; %#ok<AGROW>
        results(idx).maxit_capped_by_memory = capped; %#ok<AGROW>
    end
end
