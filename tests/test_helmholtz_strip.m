function tests = test_helmholtz_strip
%TEST_HELMHOLTZ_STRIP The shifted strip operator and its Schur complement.
%   Covers the Helmholtz path added for the helmholtz_1d family: the sine
%   transform still inverts the half-domain once a constant shift is folded
%   into the mode eigenvalues, the resonance diagnostic reports the true
%   distance to singularity, and both the fast and the generic path produce
%   the same Schur complement. The generic path matters because it cannot
%   use Cholesky here: the shifted block is indefinite.
tests = functiontests(localfunctions);
end

function testShiftedSolverIsExact(testCase)
[mx, ny] = deal(25, 96);
[A, shift] = strip(mx, ny, 32);
solve = dst_strip_solver(mx, ny, shift);
rng(0); B = randn(mx * ny, 3);
verifyLessThan(testCase, norm(A * solve(B) - B, 'fro') / norm(B, 'fro'), 1e-12);
end

function testZeroShiftMatchesLaplacian(testCase)
[mx, ny] = deal(25, 64);
rng(0); B = randn(mx * ny, 2);
shifted = dst_strip_solver(mx, ny, 0);
plain   = dst_strip_solver(mx, ny);
verifyEqual(testCase, shifted(B), plain(B), 'AbsTol', 0, 'RelTol', 0);
end

function testResonanceInfoMatchesTrueSpectrum(testCase)
[mx, ny] = deal(25, 96);
for ppw = [32 16]
    [A, shift] = strip(mx, ny, ppw);
    [~, info] = dst_strip_solver(mx, ny, shift);
    verifyEqual(testCase, info.min_abs_eig, min(abs(eig(full(A)))), 'RelTol', 1e-10);
end
end

function testShiftedOperatorIsIndefinite(testCase)
% If it were not, the experiment would not be testing anything Poisson does
% not already cover, and the Cholesky path would have sufficed.
[mx, ny] = deal(25, 96);
[A, ~] = strip(mx, ny, 32);
verifyGreaterThan(testCase, nnz(eig(full(A)) < 0), 0);
end

function testSchurAgreesWithDenseReference(testCase)
[mx, ny] = deal(25, 64);
nx = 2 * mx + 1;
[A, shift] = strip(nx, ny, 32);
I1 = 1:(mx * ny);
I2 = I1 + ((mx + 1) * ny);
I3 = mx * ny + (1:ny);
Sref = full(A(I3, I3) - A(I3, I1) * (A(I1, I1) \ full(A(I1, I3))) ...
                      - A(I3, I2) * (A(I2, I2) \ full(A(I2, I3))));
rng(1); X = randn(ny, 3);
for dims = {[mx ny], []}                 % fast path, then generic fallback
    mv = form_S(A, I1, I2, I3, dims{1}, [], shift);
    verifyLessThan(testCase, ...
        norm(mv(X) - Sref * X, 'fro') / norm(Sref * X, 'fro'), 1e-11);
end
end

% ------------------------------------------------------------------------
function [A, shift] = strip(nx, ny, ppw)
h = 1 / (ny + 1);
k = 2 * pi / (ppw * h);
A = fd_helmholtz_strip(nx, ny, k);
shift = (h * k)^2;
end
