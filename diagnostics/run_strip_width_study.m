function run_strip_width_study()
%RUN_STRIP_WIDTH_STUDY Cost and ranks of the Helmholtz strip against its width.
%   The question (2026-10-09, Section 5.3 of the paper): the transverse width
%   n_x = 51 was fixed when the Schur complement was applied by a sparse
%   solve; with the DST-based fast operator, is the constraint still
%   needed? This times one matvec of the fast operator at widths 51 to 801
%   for the three Table 5 sizes, and builds the factorization at N = 4096,
%   eps = 1e-6, at widths 51 to 401 (kernel_params.strip_width).
%
%   Results (laptop, multithreaded, seed 0, 2026-10-09):
%     matvec time grows linearly with the width, 0.0020 s (51) to 0.035 s
%     (801) at N = 4096 and 0.0079 s to 0.138 s at N = 16384, so the 288
%     matvecs of a Table 5 run cost 0.6 s to 40 s: affordable at any width.
%     Factorization at N = 4096, eps = 1e-6:
%       nx= 51 ( 1.6 wavelengths,  3 propagating modes): ranks 14/14/14/13, n_mv 288, relres 5.9e-5, unprec 1370
%       nx=101 ( 3.2 wavelengths,  6 modes):              ranks 18/20/25/24, n_mv 304, relres 9.7e-5, unprec  767
%       nx=201 ( 6.3 wavelengths, 12 modes):              ranks 26/26/39/16, n_mv 304, relres 4.6e-6, unprec  763
%       nx=401 (12.5 wavelengths, 25 modes):              ranks 34/40/46/30, n_mv 304, relres 1.3e-4, unprec 1096
%     The ranks track the number of propagating transverse modes, and the
%     resonance density grows with the width (n_x N resonances), which is
%     the real cost of a wider strip, not the matvec.
addpath(genpath(fileparts(fileparts(mfilename('fullpath')))));  % repository root
widths = [51 101 201 401 801]; sizes = [4096 8192 16384];
fprintf('--- matvec time (median of 5), fast operator; x 288 = the matvec budget of one Table 5 run\n');
for ny = sizes
  for nx = widths
    [~, b, ~, ~, mv, ~] = evalc_quiet(ny, nx);
    v = randn(ny, 1); mv(v); t = zeros(1,5);
    for r = 1:5, t0 = tic; mv(v); t(r) = toc(t0); end
    fprintf('MV  N=%5d nx=%3d (%.1f wavelengths across): %.4f s per matvec, x288 = %.1f s\n', ny, nx, nx/32, median(t), 288*median(t));
  end
end
fprintf('--- factorization probe at N=4096, eps=1e-6 (local, multithreaded)\n');
for nx = widths(1:4)
  params = struct('dim',1,'kernel_type','helmholtz_schur','kernel_params',struct('points_per_wavelength',32,'strip_width',nx), ...
    'user_tolerance',1e-6,'step_b',16,'num_levels',round(log2(4096/256))+1,'pt_leaf',256,'method','householder', ...
    'sample_distribution','auto','compute_cond',false,'warmup',false,'collect_diagnostics',false, ...
    'operator_probe_count',0,'comparison_seed',0,'unpreconditioned_baseline',true);
  rng(0,'twister'); t0 = tic;
  try
    [~, st] = main(params);
    fprintf('FACT nx=%3d: level_ranks=%s n_mv=%d t_fact=%.2f relres=%.2e n_it=%d unprecond=%d(flag %d) wall=%.0fs\n', ...
      nx, st.level_ranks, st.n_matvec, st.t_factor, st.solve_residual, st.krylov_iterations, st.unpreconditioned_iterations, st.unpreconditioned_solver_flag, toc(t0));
  catch e
    fprintf('FACT nx=%3d: FAILED %s\n', nx, e.message);
  end
end
end
function varargout = evalc_quiet(ny, nx)
  [~, varargout{1:6}] = evalc('build_test_system(ny, 256, ''helmholtz_schur'', 1, struct(''points_per_wavelength'', 32, ''strip_width'', nx), ''real'')');
end
