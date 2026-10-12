function solve = dst_poisson_solver(mx, my, mz)
%DST_POISSON_SOLVER Fast direct solver for a Dirichlet Laplacian on a box.
%   SOLVE = DST_POISSON_SOLVER(MX, MY, MZ) returns a function handle with
%   Z = SOLVE(B) solving L*Z = B, where L is the (MX*MY*MZ)-by-(MX*MY*MZ)
%   seven-point Dirichlet Laplacian
%
%       L = kron(Lx, kron(Iy, Iz)) + kron(Ix, kron(Ly, Iz))
%           + kron(Ix, kron(Iy, Lz)),      Lm = tridiag(-1, 2, -1)
%
%   built exactly as in POISSON_3D, with z the fastest-varying index.
%
%   WHY THIS IS APPLICABLE HERE, AND ONLY HERE
%   ------------------------------------------
%   This is an optimization of the *test harness*, not of the HSS algorithm.
%   It exploits four properties that our synthetic test problems happen to
%   have and that a general input to the HSS solver does not:
%
%     1. Separability. The operator is a Kronecker sum of one-dimensional
%        operators, so it diagonalizes in the tensor product of the 1D
%        eigenbases. This fails for variable coefficients, for any operator
%        with cross terms, and for non-Cartesian geometry.
%     2. A uniform Cartesian grid, so every 1D factor is the *same* constant
%        coefficient tridiagonal matrix and its eigenvectors are known in
%        closed form.
%     3. Homogeneous Dirichlet conditions on every face, which is what makes
%        the eigenvectors sines. The half-domains A(I1,I1) and A(I2,I2) of
%        POISSON_3D qualify: each is bounded by the physical boundary on one
%        side and by the separator plane on the other, and the separator
%        columns are excluded from the block, which is exactly a homogeneous
%        Dirichlet condition there.
%     4. A constant diagonal. A nugget or shift sigma^2*I is still fine (it
%        shifts every eigenvalue), but a spatially varying diagonal is not.
%
%   The eigenvalues of the m-by-m tridiag(-1,2,-1) with Dirichlet ends are
%   lambda_p = 2 - 2*cos(p*pi/(m+1)), and its normalized eigenvectors form
%   the DST-I matrix S(p,q) = sqrt(2/(m+1))*sin(p*q*pi/(m+1)), which is
%   symmetric and orthogonal, so S is its own inverse and the forward and
%   backward transforms are the same operation.
%
%   Cost is O(N*(mx+my+mz)) per right-hand side using dense transforms,
%   against O(N^{4/3}) memory and a substantial factorization for sparse
%   Cholesky, and it needs no factorization or fill-in at all.
%
%   See also POISSON_3D, FORM_S_2D.

    validateattributes(mx, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(my, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(mz, {'numeric'}, {'scalar', 'integer', 'positive'});

    Sx = sine_matrix(mx);
    Sy = sine_matrix(my);
    Sz = sine_matrix(mz);

    % Lambda(k,j,i) = lz(k) + ly(j) + lx(i), matching the (z,y,x) layout.
    lx = dirichlet_eigenvalues(mx);
    ly = dirichlet_eigenvalues(my);
    lz = dirichlet_eigenvalues(mz);
    Lambda = reshape(lz, [], 1) + reshape(ly, 1, []) + reshape(lx, 1, 1, []);

    solve = @apply_solve;

    function Z = apply_solve(B)
        r = size(B, 2);
        X = reshape(B, mz, my, mx, r);
        X = mode_multiply(Sz, X, 1);
        X = mode_multiply(Sy, X, 2);
        X = mode_multiply(Sx, X, 3);
        X = X ./ Lambda;                      % diagonal solve in the sine basis
        X = mode_multiply(Sx, X, 3);          % S is orthogonal and symmetric,
        X = mode_multiply(Sy, X, 2);          % so the inverse transform is S
        X = mode_multiply(Sz, X, 1);
        Z = reshape(X, mz * my * mx, r);
    end
end

% ------------------------------------------------------------------------
function S = sine_matrix(m)
%SINE_MATRIX Orthogonal DST-I matrix; S = S' = inv(S).
    p = (1:m)';
    S = sqrt(2 / (m + 1)) * sin(p * p' * pi / (m + 1));
end

% ------------------------------------------------------------------------
function lam = dirichlet_eigenvalues(m)
%DIRICHLET_EIGENVALUES Eigenvalues of tridiag(-1, 2, -1) of order m.
    lam = 2 - 2 * cos((1:m) * pi / (m + 1));
end

% ------------------------------------------------------------------------
function Y = mode_multiply(S, X, d)
%MODE_MULTIPLY Apply S along dimension d of the array X.
    p = 1:ndims(X);
    p([1 d]) = p([d 1]);                      % a transposition, so p undoes itself
    Xp = permute(X, p);
    sz = size(Xp);
    Y = permute(reshape(S * reshape(Xp, sz(1), []), sz), p);
end
