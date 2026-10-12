function out = plot_cavity_fields(n1d, out_dir, ppw)
%PLOT_CAVITY_FIELDS Scattering potential and total field of the cavity problem.
%   OUT = PLOT_CAVITY_FIELDS(N1D, OUT_DIR) solves the Lippmann-Schwinger
%   equation of the manuscript's Section 5.5 for the plane wave
%   u0(x) = exp(i k x1) incident on the cavity potential of Gopal and
%   Martinsson (arXiv:2007.12718, Sec. 6.2), on the same n1d-by-n1d grid and
%   with the same discretization as build_test_system.m (kind 'helmholtz',
%   potential 'cavity': k = 2*pi*n1d/32, i.e. 32 points per wavelength,
%   piecewise-constant collocation, block-Toeplitz Green's matrix applied
%   by a 2D FFT), and writes to OUT_DIR
%       potential.pdf        eta(x)
%       total_field_abs.pdf  |u0 + us|
%       total_field_re.pdf   Re(u0 + us)
%   with N1D and the number of wavelengths q in the file names.
%
%   Convention (Section 5.5): sigma + w^2 eta int G sigma = -w^2 eta u0,
%   u = u0 + int G sigma. With mu = sigma/sqrt(eta) and sqrtb = k sqrt(eta),
%   the system is (I + sqrtb G sqrtb) mu = -k sqrtb u0, sigma = sqrtb mu / k.
%   The solve uses unrestarted GMRES at 1e-10 on the bare operator; no
%   factorization is needed for a picture.

    if nargin < 1 || isempty(n1d), n1d = 256; end
    if nargin < 2 || isempty(out_dir)
        out_dir = fullfile(fileparts(mfilename('fullpath')), 'comparison_results', 'cavity_fields');
    end
    % PPW (optional, default 32 as in the experiments) sets the points per
    % wavelength, so a resolution check at fixed q is plot_cavity_fields(2*n1d, dir, 64).
    if nargin < 3 || isempty(ppw), ppw = 32; end
    if ~isfolder(out_dir), mkdir(out_dir); end

    q = n1d / ppw;
    k = 2 * pi * q;
    h = 1 / n1d;
    xx = linspace(h, 1, n1d);
    [X, Y] = meshgrid(xx, xx);          % X(i,j) = xx(j) is x1, Y(i,j) = xx(i) is x2
    xc = X(:) - 0.5; yc = Y(:) - 0.5;
    r = sqrt(xc.^2 + yc.^2); theta = atan2(yc, xc);
    V = (1 - sin(0.5 * theta).^500) .* exp(-2000 * (0.1 - r.^2).^2);
    sqrtb = k * sqrt(V);
    u0 = exp(1i * k * X(:));

    % Green's matrix, block Toeplitz, embedded in a circulant (as in build_test_system.m)
    grid_vec = (0:n1d-1) * h;
    [xd, yd] = ndgrid(grid_vec, grid_vec);
    t = (1i/4) * besselh(0, 1, k * sqrt(xd.^2 + yd.^2)) * h^2;
    t(1,1) = integral2(@(x,y) 0.25i * besselh(0, 1, k * sqrt(x.^2 + y.^2)), -h/2, h/2, -h/2, h/2);
    B = zeros(2*n1d-1, 'like', t);
    B(1:n1d, 1:n1d) = t;
    B(1:n1d, n1d+1:end) = t(:, 2:end);
    B(n1d+1:end, 1:n1d) = t(2:end, :);
    B(n1d+1:end, n1d+1:end) = t(2:end, 2:end);
    B(n1d+1:end, :) = flip(B(n1d+1:end, :), 1);
    B(:, n1d+1:end) = flip(B(:, n1d+1:end), 2);
    Ghat = fft2(B);
    function y = conv_G(w)
        P = zeros(2*n1d-1, 'like', Ghat);
        P(1:n1d, 1:n1d) = reshape(w, n1d, n1d);
        C = ifft2(Ghat .* fft2(P));
        y = reshape(C(1:n1d, 1:n1d), [], 1);
    end
    mv = @(x) x + sqrtb .* conv_G(sqrtb .* x);

    f = -k * sqrtb .* u0;
    timer = tic;
    [mu, flag, relres, iter] = gmres(mv, f, [], 1e-10, 2000);
    fprintf('n1d=%d (q=%g): GMRES flag %d, %d iterations, relres %.2e, %.1fs\n', ...
        n1d, q, flag, iter(2), relres, toc(timer));
    sigma = sqrtb .* mu / k;
    u = u0 + conv_G(sigma);

    tag = sprintf('N%d_q%g', n1d, q);
    if ppw ~= 32, tag = sprintf('%s_ppw%d', tag, ppw); end
    write_panel(reshape(V, n1d, n1d), xx, 'parula', [0 1], fullfile(out_dir, ['potential_' tag '.pdf']));
    write_panel(reshape(abs(u), n1d, n1d), xx, 'parula', [], fullfile(out_dir, ['total_field_abs_' tag '.pdf']));
    m = max(abs(real(u)));
    write_panel(reshape(real(u), n1d, n1d), xx, 'redblue', [-m m], fullfile(out_dir, ['total_field_re_' tag '.pdf']));

    out = struct('n1d', n1d, 'q', q, 'gmres_iterations', iter(2), 'gmres_flag', flag, ...
        'max_abs_u', max(abs(u)), 'out_dir', out_dir);
end

function write_panel(M, xx, cmap, clim, file)
% Each panel is drawn at the width it is printed at in the manuscript
% (0.47 of a 338 pt text width, 2.2 in, two panels per row) with 10 pt
% tick and colorbar fonts at 7 pt and no axis labels, so nothing is
% scaled down on the page. Exported as PDF so the text stays vector.
    fig = figure('Visible', 'off', 'Color', 'w', 'Units', 'inches', ...
        'Position', [1 1 2.25 2.05]);
    ax = axes(fig);
    % Grid points sit at x = h, 2h, ..., 1; draw each pixel centered on
    % its point so the image spans exactly [0, 1] and the 0 tick is drawn.
    h = xx(2) - xx(1);
    imagesc(ax, xx - h/2, xx - h/2, M);
    axis(ax, 'xy', 'image');
    xlim(ax, [0 1]); ylim(ax, [0 1]);
    if ~isempty(clim), caxis(ax, clim); end
    if ischar(cmap) && strcmp(cmap, 'redblue')
        n = 256; s = linspace(0, 1, n/2)';
        colormap(ax, [[s, s, ones(n/2,1)]; [ones(n/2,1), flipud(s), flipud(s)]]);
    else
        colormap(ax, cmap);
    end
    % Manuscript layout (2026-10-09): no axis labels (the caption names the axes), and
    % tick and colorbar fonts 30 percent below the 10 pt body size.
    cb = colorbar(ax);
    cb.FontSize = 7; cb.TickLabelInterpreter = 'latex';
    xticks(ax, 0:0.5:1); yticks(ax, 0:0.5:1);
    set(ax, 'FontSize', 7, 'TickLabelInterpreter', 'latex', 'TickDir', 'out', 'LineWidth', 0.5);
    exportgraphics(fig, file, 'ContentType', 'vector');
    close(fig);
end
