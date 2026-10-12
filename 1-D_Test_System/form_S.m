% For Poisson kernel

function mvS = form_S(A, I1, I2, I3, dst_dims, mode, shift)
% Extracts blocks from the 2-D discretization matrix A (nx*ny x nx*ny)
% and computes the Schur complement:
%
%   S = A_33 - A_31*A_11^{-1}*A_13 - A_32*A_22^{-1}*A_23
%
% Inputs:
%   A           - The nx*ny x nx*ny discretization matrix.
%   I1, I2, I3  - Partitioned indices
%   dst_dims    - Optional [mx ny]. Solve with the half-domain blocks
%                 through DST_STRIP_SOLVER instead of a sparse Cholesky
%                 factorization. Valid only when the blocks are
%                 constant-coefficient Dirichlet Laplacians on a uniform
%                 mx-by-ny strip, as in POISSON_2D(51, n) with mx = 25. The
%                 solver is verified on a probe before it is trusted and the
%                 factorization is skipped entirely when it passes, which is
%                 what makes n = 2^21 affordable. Omit or pass [] for the
%                 factorization. Test-harness optimization only: the HSS
%                 algorithm still sees S through matrix-vector products.
%
%   shift       - Optional spectral shift of the half-domain blocks, so that
%                 they are (L - shift*I) rather than L. Pass (h*k)^2 for the
%                 Helmholtz strip of FD_HELMHOLTZ_STRIP; omit or pass 0 for
%                 Poisson. A nonzero shift makes the blocks symmetric
%                 indefinite, so the generic fallback factors them with LDL
%                 rather than Cholesky, which would fail.
%
% Output:
%   S - The resulting block Schur complement.

    fprintf("At form_S...\n");
    if nargin < 5
        dst_dims = [];
    end
    if nargin < 6 || isempty(mode)
        mode = 'reduced';   % or 'strip': strip solver on the full blocks
    end
    if nargin < 7 || isempty(shift)
        shift = 0;
    end

    % ---- Extract the corresponding blocks from A ----
    A11 = A(I1, I1);
    A13 = A(I1, I3);
    A31 = A(I3, I1);
    A33 = A(I3, I3);
    
    A22 = A(I2, I2);
    A23 = A(I2, I3);
    A32 = A(I3, I2);

    % ---- Optional fast path: sine transform along x, tridiagonal solves ----
    % Verified on a probe so wrong dimensions degrade to the factorization
    % instead of returning wrong answers silently. Both half-domains are the
    % same strip, so one solver serves A11 and A22; each is probed.
    use_dst = false;
    if ~isempty(dst_dims)
        if prod(dst_dims) ~= size(A11, 1)
            error('form_S:dstDims', ...
                'dst_dims product %d does not match the block size %d.', ...
                prod(dst_dims), size(A11, 1));
        end
        mx = dst_dims(1);
        ny = dst_dims(2);
        [dst_solve, dst_info] = dst_strip_solver(mx, ny, shift);
        if shift ~= 0
            fprintf("  - Shifted strip: min |eig| = %.3e (relative %.3e).\n", ...
                dst_info.min_abs_eig, dst_info.resonance_ratio);
            if dst_info.resonance_ratio < 1e-6
                warning('form_S:nearResonance', ...
                    ['The shifted half-domain is within %.3e of singular ' ...
                     '(relative %.3e). The Schur complement it defines is ' ...
                     'dominated by that near-null mode; change the ' ...
                     'wavenumber or the grid.'], ...
                    dst_info.min_abs_eig, dst_info.resonance_ratio);
            end
        end
        % Probes come from a private stream so that building the operator
        % leaves the global stream, and hence the sketch Omega drawn after it,
        % exactly as the factorization path does. At N = 262144, eps = 1e-9,
        % a probe drawn from the global stream shifted Omega and the leaves
        % of that particular draw failed the stopping test at full rank.
        probe_stream = RandStream('mt19937ar', 'Seed', 20260913);
        probe = randn(probe_stream, size(A11, 1), 2);
        rel1 = norm(A11 * dst_solve(probe) - probe, 'fro') / norm(probe, 'fro');
        rel2 = norm(A22 * dst_solve(probe) - probe, 'fro') / norm(probe, 'fro');
        if max(rel1, rel2) < 1e-10
            use_dst = true;
            fprintf("  - DST strip solver verified (residuals %.1e, %.1e); using it.\n", rel1, rel2);
        else
            warning("DST strip solver residuals %.1e, %.1e are too large; " + ...
                "falling back to the Cholesky factorization.", rel1, rel2);
        end
    end

    % Reduced form. The separator couples to one x-slice of each strip
    % through a multiple of the identity, A13 = c1 * (e_j1 (x) I_ny), so in
    % the sine basis along x the correction A31*A11^{-1}*A13 is
    %     sum_i c1^2 * Sx(j1,i)^2 * (Ly + lambda_i I)^{-1},
    % and likewise for the other strip: ny-by-ny tridiagonal solves only,
    % never a 25*ny-by-r intermediate. Measured at ny = 131072 the generic
    % strip solve of one 16-column block moves about 3 GB through memory
    % and takes 1.7 s; this takes 0.2 s. The structure is checked and the
    % result is compared with the generic form on a probe before it is used.
    use_reduced = false;
    if use_dst && strcmpi(mode, 'reduced')
        [ok1, j1, c1] = separator_coupling(A13, mx, ny);
        [ok2, j2, c2] = separator_coupling(A23, mx, ny);
        if ok1 && ok2 && isequal(A31, A13.') && isequal(A32, A23.')
            pidx = (1:mx)';
            Sx = sqrt(2 / (mx + 1)) * sin(pidx * pidx' * pi / (mx + 1));
            lambda = 2 - 2 * cos((1:mx) * pi / (mx + 1)) - shift;
            weights = c1^2 * Sx(j1, :).^2 + c2^2 * Sx(j2, :).^2;
            Ly = spdiags(repmat([-1 2 -1], ny, 1), -1:1, ny, ny);
            T = cell(mx, 1);
            for i = 1:mx
                T{i} = Ly + lambda(i) * speye(ny);
            end
            Xp = randn(probe_stream, ny, 2);
            ref = A33 * Xp - A31 * dst_solve(A13 * Xp) - A32 * dst_solve(A23 * Xp);
            relr = norm(apply_S_reduced(Xp) - ref, 'fro') / norm(ref, 'fro');
            if relr < 1e-10
                use_reduced = true;
                fprintf("  - Reduced separator form verified (residual %.1e); using it.\n", relr);
            else
                warning("Reduced separator form residual %.1e is too large; " + ...
                    "using the strip solver on the full blocks.", relr);
            end
        end
    end

    if use_reduced
        mvS = @apply_S_reduced;
        return;
    elseif use_dst
        mvS = @apply_S_dst;
        return;
    end

    % ---- Perform matrix multiplication using pivoted Cholesky ----
    % S * X = A33 * X - A31 * (A11 \ (A13 * X)) - A32 * (A22 \ (A23 * X )));

    % -- A11, A22 --
    % Cholesky only when the block is positive definite. A Helmholtz shift
    % large enough to make the strip indefinite -- which is the interesting
    % regime, and the one the experiments use -- makes chol error out, so the
    % shifted case factors with DECOMPOSITION, which selects an LDL for a
    % sparse symmetric indefinite matrix. The separator splits the grid
    % symmetrically, so A22 is usually the same matrix as A11; reuse the
    % factor rather than building it twice.
    if shift == 0
        fprintf("  - Factorizing A11 (sparse Cholesky)...\n");
        [L1, ~, P1] = chol(A11, 'lower', 'vector');
        solve_A11 = @(M) chol_solve(L1, P1, M);
        if isequal(A11, A22)
            solve_A22 = solve_A11;
        else
            [L2, ~, P2] = chol(A22, 'lower', 'vector');
            solve_A22 = @(M) chol_solve(L2, P2, M);
        end
    else
        fprintf("  - Factorizing A11 (sparse LDL, shifted block)...\n");
        D1 = decomposition(A11, 'ldl');
        solve_A11 = @(M) D1 \ M;
        if isequal(A11, A22)
            solve_A22 = solve_A11;
        else
            D2 = decomposition(A22, 'ldl');
            solve_A22 = @(M) D2 \ M;
        end
    end

    mvS = @apply_S;

    function Y = apply_S(X)
        % Computes Y = S * X without forming S explicitly.
        %   S * X = A33*X
        %         - A31*[ A11^{-1} * (A13*X) ]
        %         - A32*[ A22^{-1} * (A23*X) ].

        Y = A33 * X;

        T1 = A13 * X;            
        Z1 = solve_A11(T1);       
        Y = Y - A31 * Z1; 

        T2 = A23 * X;
        Z2 = solve_A22(T2);
        Y = Y - A32 * Z2;       
    end

    function Y = apply_S_dst(X)
        % Same Schur complement, half-domain solves through the strip solver.
        Y = A33 * X - A31 * dst_solve(A13 * X) - A32 * dst_solve(A23 * X);
    end

    function Y = apply_S_reduced(X)
        % S*X = A33*X - sum_i weights(i) * (Ly + lambda_i I)^{-1} X.
        Y = A33 * X;
        for i = 1:mx
            Y = Y - weights(i) * (T{i} \ X);
        end
    end
end

function Y = chol_solve(L, P, M)
%CHOL_SOLVE Apply the inverse through a permuted sparse Cholesky factor.
    Yperm = L' \ (L \ M(P, :));
    Y = zeros(size(M), 'like', M);
    Y(P, :) = Yperm;
end

function [ok, j, c] = separator_coupling(A13, mx, ny)
%SEPARATOR_COUPLING Is A13 = c * (e_j (x) I_ny) for one slice j and a scalar c?
    ok = false; j = 0; c = 0;
    if size(A13, 1) ~= mx * ny || size(A13, 2) ~= ny || nnz(A13) ~= ny
        return;
    end
    [rows, cols, vals] = find(A13);
    slice = ceil(rows / ny);
    if any(slice ~= slice(1)) || any(rows - (slice(1) - 1) * ny ~= cols) ...
            || any(abs(vals - vals(1)) > 0)
        return;
    end
    ok = true; j = slice(1); c = vals(1);
end
