function mvS = form_S_2D(A, I1, I2, I3, dst_dims)
% Computes the Schur complement operator:
%   S = A_33 - A_31*A_11^{-1}*A_13 - A_32*A_22^{-1}*A_23
%
% DST_DIMS is optional. Pass [mx my mz] to solve with the half-domain blocks
% using a discrete sine transform instead of a sparse factorization. This is
% ONLY valid when those blocks are constant-coefficient Dirichlet Laplacians
% on a uniform mx-by-my-by-mz box, which holds for the Poisson test problem
% and NOT for the Helmholtz one; see DST_POISSON_SOLVER for the full list of
% assumptions. It is an optimization of the test harness only: the HSS
% algorithm still sees A through matrix-vector products alone. Omit or pass
% [] to use the factorization path.

    fprintf("At form_S_2D ...\n");

    % Extract the corresponding blocks from A
    A11 = A(I1, I1);
    A13 = A(I1, I3);
    A31 = A(I3, I1);
    A33 = A(I3, I3);
    
    A22 = A(I2, I2);
    A23 = A(I2, I3);
    A32 = A(I3, I2);
    
    % Let decomposition pick the factorization. For the Poisson case A11 is
    % SPD and it selects Cholesky, whose solves are more than an order of
    % magnitude faster than the LDL path that was forced here; for indefinite
    % cases (Helmholtz) it falls back to LDL on its own.
    fprintf("  - Precomputing factorization for A11...\n");
    dA11 = decomposition(A11);

    % The separator plane splits the grid symmetrically, so the two halves are
    % the same matrix. Factor once and reuse.
    if isequal(A11, A22)
        fprintf("  - A22 is identical to A11; reusing its factorization.\n");
        dA22 = dA11;
    else
        fprintf("  - Precomputing factorization for A22...\n");
        dA22 = decomposition(A22);
    end

    fprintf("  - Factorizations done (%s).\n", dA11.Type);

    % Optional DST path. Verify it actually inverts A11 before trusting it,
    % so a caller passing the wrong box dimensions degrades to the
    % factorization instead of silently returning wrong answers.
    use_dst = false;
    if nargin >= 5 && ~isempty(dst_dims)
        if prod(dst_dims) ~= size(A11, 1)
            error('form_S_2D:dstDims', ...
                'dst_dims product %d does not match the block size %d.', ...
                prod(dst_dims), size(A11, 1));
        end
        dst_solve = dst_poisson_solver(dst_dims(1), dst_dims(2), dst_dims(3));
        probe = randn(size(A11, 1), 2);
        rel = norm(A11 * dst_solve(probe) - probe, 'fro') / norm(probe, 'fro');
        if rel < 1e-10
            use_dst = true;
            fprintf("  - DST solver verified (residual %.1e); using it.\n", rel);
        else
            warning("DST solver residual %.1e is too large; " + ...
                "falling back to the factorization.", rel);
        end
    end

    mvS = @apply_S;

    function Y = apply_S(X)
        % Computes Y = S * X
        
        Y = A33 * X;

        % A31 * (A11 \ (A13 * X)) and A32 * (A22 \ (A23 * X)).
        % A11 and A22 are the same matrix here, so one solver serves both.
        T1 = A13 * X;
        T2 = A23 * X;
        if use_dst
            Z1 = dst_solve(T1);
            Z2 = dst_solve(T2);
        else
            Z1 = dA11 \ T1;
            Z2 = dA22 \ T2;
        end
        Y = Y - A31 * Z1 - A32 * Z2;
    end
end