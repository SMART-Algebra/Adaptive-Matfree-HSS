function run_poisson_width_study()
%RUN_POISSON_WIDTH_STUDY Cost and ranks of the Poisson Schur complements against the strip width.
%   Companion of run_strip_width_study.m for the two Poisson problems
%   (kernel_params.strip_width; default 51 as in the manuscript). Times one
%   matvec of the fast operator and builds the factorization at eps = 1e-6.
%
%   Results (laptop, multithreaded, seed 0, 2026-10-09):
%   1D Poisson Schur (2D strip), matvec x 288:
%     N=8192:   1.0 / 1.9 / 3.9 / 7.6 / 15.7 s at nx = 51 / 101 / 201 / 401 / 801
%     N=2^17:  15.6 / 31 / 61 / 124 / 241 s             (linear in the width)
%   1D factorization at N=8192: max rank 15 / 23 / 20 / 26 / 33, n_mv 288 -> 304,
%     relres 1.1e-6 / 3.5e-6 / 6.9e-6 / 1.6e-5 / 4.3e-5, n_it 3 throughout:
%     the ranks grow slowly (about logarithmically) with the width as the
%     strip's Schur complement approaches the half-plane map; the residual
%     grows with it.
%   2D Poisson Schur (3D slab), matvec x 1952 (n_mv at 128^2, 1e-6):
%     N=128^2: 37 s at nx = 51, 85 s at nx = 101 (width 201 exhausted the
%     laptop's memory while building the 3.3-million-unknown sparse matrix).
%   2D factorization at N=128^2: nx = 51: ranks 177/384/416, n_mv 1952,
%     298 MB, relres 3.1e-5; nx = 101: ranks 177/409/416, n_mv 2052, 312 MB,
%     relres 3.5e-5. The slab thickness is immaterial to the 2D interface
%     operator at these sizes; only the matvec cost changes.
%
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));  % repository root
fprintf('--- 1D Poisson Schur (2D strip): matvec time x288\n');
for n = [8192 131072]
  for nx = [51 101 201 401 801]
    [~, ~, ~, ~, mv, ~] = evalc_quiet(n, 1, nx);
    v = randn(n,1); mv(v); t = zeros(1,5); for r = 1:5, t0 = tic; mv(v); t(r) = toc(t0); end
    fprintf('MV1  N=%6d nx=%3d: %.4f s per matvec, x288 = %.1f s\n', n, nx, median(t), 288*median(t));
  end
end
fprintf('--- 2D Poisson Schur (3D slab): matvec time x1952 (n_mv at 128^2, 1e-6)\n');
for n1d = [128 256]
  for nx = [51 101 201]
    [~, ~, ~, ~, mv, ~] = evalc_quiet(n1d, 2, nx);
    v = randn(n1d^2,1); mv(v); t = zeros(1,3); for r = 1:3, t0 = tic; mv(v); t(r) = toc(t0); end
    fprintf('MV2  N=%3d^2 nx=%3d: %.4f s per matvec, x1952 = %.0f s\n', n1d, nx, median(t), 1952*median(t));
  end
end
fprintf('--- factorization probes, eps=1e-6 (see run_poisson_width_study header)\n');
for nx = [51 101 201 401 801]
  params = base(1, 8192, nx); rng(0,'twister'); t0 = tic;
  [~, st] = main(params);
  fprintf('FACT1 N=8192  nx=%3d: level_ranks=%s n_mv=%d t_fact=%.2f relres=%.2e n_it=%d wall=%.0fs\n', nx, st.level_ranks, st.n_matvec, st.t_factor, st.solve_residual, st.krylov_iterations, toc(t0));
end
for nx = [51 101 201]
  params = base(2, 128, nx); rng(0,'twister'); t0 = tic;
  [~, st] = main(params);
  fprintf('FACT2 N=128^2 nx=%3d: level_ranks=%s n_mv=%d t_fact=%.1f relres=%.2e mem=%.0fMB n_it=%d wall=%.0fs\n', nx, st.level_ranks, st.n_matvec, st.t_factor, st.solve_residual, st.factor_storage_mb, st.krylov_iterations, toc(t0));
end
end
function p = base(dim, n, nx)
  if dim == 1, levels = round(log2(n/256))+1; b = 16; else, levels = round(log2(n/16))+1; b = 32; end
  p = struct('dim',dim,'kernel_type','poisson_schur','kernel_params',struct('strip_width',nx), ...
    'user_tolerance',1e-6,'step_b',b,'num_levels',levels,'pt_leaf',256,'method','householder', ...
    'sample_distribution','auto','compute_cond',false,'warmup',false,'collect_diagnostics',false, ...
    'operator_probe_count',0,'comparison_seed',0,'unpreconditioned_baseline',false);
end
function varargout = evalc_quiet(n, dim, nx)
  [~, varargout{1:6}] = evalc(sprintf('build_test_system(%d, 256, ''poisson_schur'', %d, struct(''strip_width'', %d), ''real'')', n, dim, nx));
end
