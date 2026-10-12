function [solve, info] = dst_strip_solver(mx, ny, shift)
%DST_STRIP_SOLVER Fast direct solver for a Dirichlet Laplacian on a strip.
%   SOLVE = DST_STRIP_SOLVER(MX, NY) returns a function handle with
%   Z = SOLVE(B) solving L*Z = B, where L is the (MX*NY)-by-(MX*NY)
%   five-point Dirichlet Laplacian
%
%       L = kron(Lx, Iy) + kron(Ix, Ly),      Lm = tridiag(-1, 2, -1),
%
%   built exactly as in POISSON_2D, with y the fastest-varying index. The
%   half-domains A(I1,I1) and A(I2,I2) of POISSON_2D(51, n) are such strips
%   with MX = 25 and NY = n.
%
%   The strip is short in x and very long in y (n reaches 2^21 in the
%   paper), so only the x direction is diagonalized: the MX-by-MX DST-I
%   matrix S, symmetric and orthogonal, turns L into MX decoupled
%   tridiagonal systems (Ly + lambda_i I) along y, which are solved
%   together as one tridiagonal SPD system of order MX*NY. A dense NY-by-NY
%   sine matrix is never formed. Cost per right-hand side is
%   O(MX^2*NY) for the two transforms plus O(MX*NY) for the tridiagonal
%   solves, against a sparse Cholesky factor of the strip with about
%   25*MX*NY nonzeros and a memory-bound triangular solve per column.
%
%   Same assumptions as DST_POISSON_SOLVER: separable constant-coefficient
%   operator, uniform grid, homogeneous Dirichlet conditions on every side
%   (the separator column excluded from the block supplies the one on the
%   interior side), constant diagonal. This is an optimization of the test
%   harness only; the HSS algorithm still sees the operator through
%   matrix-vector products alone.
%
%   SOLVE = DST_STRIP_SOLVER(MX, NY, SHIFT) solves (L - SHIFT*I)*Z = B
%   instead, the strip Helmholtz operator of FD_HELMHOLTZ_STRIP with
%   SHIFT = (h*k)^2. The sine basis diagonalizes L, so a constant shift only
%   moves the mode eigenvalues: mode i becomes (Ly + (lambda_i - SHIFT)*I).
%   The transform is unchanged; what changes is that the tridiagonal systems
%   are symmetric indefinite rather than SPD once SHIFT exceeds the smallest
%   eigenvalue, so backslash factors them with LDL rather than Cholesky.
%
%   [SOLVE, INFO] = DST_STRIP_SOLVER(...) also returns INFO with fields
%   min_abs_eig, the distance from the shifted spectrum to zero, and
%   resonance_ratio, that distance relative to the spectral width. The
%   shifted operator is singular exactly when SHIFT is an eigenvalue of L,
%   and the strip spectrum is dense in NY, so a shift chosen without care
%   can land arbitrarily close to one. Check INFO before trusting a run:
%   near a resonance the half-domain solves are ill-conditioned and the
%   Schur complement they define is not the operator you meant to test.
%
%   See also POISSON_2D, FORM_S, DST_POISSON_SOLVER, FD_HELMHOLTZ_STRIP.

    validateattributes(mx, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(ny, {'numeric'}, {'scalar', 'integer', 'positive'});
    if nargin < 3 || isempty(shift), shift = 0; end
    validateattributes(shift, {'numeric'}, {'scalar', 'real', 'finite'});

    p = (1:mx)';
    Sx = sqrt(2 / (mx + 1)) * sin(p * p' * pi / (mx + 1));   % S = S' = inv(S)
    lambda = 2 - 2 * cos((1:mx) * pi / (mx + 1)) - shift;    % shifted Lx modes

    info = resonance_info(lambda, ny);

    Ly = spdiags(repmat([-1 2 -1], ny, 1), -1:1, ny, ny);
    % One tridiagonal system per sine mode -- SPD when SHIFT is zero,
    % symmetric indefinite otherwise. They are solved with the
    % banded backslash, one block at a time: measured at ny = 131072 with 16
    % right-hand sides this takes 0.19 s against 0.34 s for a precomputed
    % sparse Cholesky of the block-diagonal system and 0.86 s for backslash
    % on it, and it stores no factor (the factor would be 2.5 GB at ny = 2^21).
    T = cell(mx, 1);
    for i = 1:mx
        T{i} = Ly + lambda(i) * speye(ny);
    end

    solve = @apply_solve;

    function Z = apply_solve(B)
        r = size(B, 2);
        % B(iy + (ix-1)*ny, k): page k is the ny-by-mx array V_k, and the sine
        % transform along x is V_k * Sx on every page, no permutation needed.
        X = pagemtimes(reshape(B, ny, mx, r), Sx);
        for i = 1:mx
            X(:, i, :) = reshape(T{i} \ reshape(X(:, i, :), ny, r), ny, 1, r);
        end
        Z = reshape(pagemtimes(X, Sx), mx * ny, r);   % S is its own inverse
    end
end

% ------------------------------------------------------------------------
function info = resonance_info(lambda, ny)
%RESONANCE_INFO How close the shifted strip spectrum comes to zero.
%   The eigenvalues are lambda_i + mu_q with mu_q = 4*sin(q*pi/(2*(ny+1)))^2
%   increasing in q, so for each mode the nearest mu is found by inverting
%   that expression rather than forming all MX*NY of them (which would be
%   52 million values at ny = 2^21).
    best = inf;
    for i = 1:numel(lambda)
        t = -lambda(i);                      % the mu that would annihilate mode i
        if t <= 0
            q = 1;
        elseif t >= 4
            q = ny;
        else
            q = (2 * (ny + 1) / pi) * asin(sqrt(t / 4));
        end
        for qq = unique(min(max(round(q) + (-1:1), 1), ny))
            mu = 4 * sin(qq * pi / (2 * (ny + 1)))^2;
            best = min(best, abs(lambda(i) + mu));
        end
    end
    width = max(lambda) + 4 - (min(lambda) + 0);
    info = struct('min_abs_eig', best, 'resonance_ratio', best / max(width, eps));
end
