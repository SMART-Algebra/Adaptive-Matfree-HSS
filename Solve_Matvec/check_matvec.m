function [rel_err, t_hss] = check_matvec(node, mv, n, num_levels, ...
        kernel_type, dim, num_probes, probe_seed, verbose)
%CHECK_MATVEC Estimate the HSS operator error on independent probes.
%   The Frobenius-norm ratio over NUM_PROBES Gaussian vectors estimates
%   ||A-A_HSS|| relative to ||A|| on a fixed, reproducible test sketch.
%   The caller's random-number-generator state is restored on return.

    if nargin < 7 || isempty(num_probes), num_probes = 1; end
    if nargin < 8 || isempty(probe_seed), probe_seed = 42; end
    if nargin < 9, verbose = true; end

    previous_rng = rng;
    restore_rng = onCleanup(@() rng(previous_rng));
    rng(probe_seed, 'twister');

    probes = randn(n, num_probes);
    if strcmpi(kernel_type, 'helmholtz')
        probes = (probes + 1i * randn(n, num_probes)) / sqrt(2);
    end

    y_exact = zeros(n, num_probes, 'like', probes);
    for probe = 1:num_probes
        y_exact(:, probe) = mv(probes(:, probe));
    end

    hss_timer = tic;
    y_hss = mat_vec(node, probes, num_levels, kernel_type, dim);
    t_hss = toc(hss_timer);
    rel_err = norm(y_exact-y_hss, 'fro') / max(norm(y_exact, 'fro'), realmin);

    if verbose
        fprintf('Randomized HSS operator error (%d probes): %.2e\n', ...
            num_probes, rel_err);
        fprintf('HSS probe-application time: %.2e s\n', t_hss);
    end
end
