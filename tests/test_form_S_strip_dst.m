function tests = test_form_S_strip_dst
%TEST_FORM_S_STRIP_DST The 1D Poisson Schur operator's fast path equals the Cholesky path.
%   The half-domains of POISSON_2D(51, n) are 25-by-n Dirichlet strips;
%   DST_STRIP_SOLVER inverts them with a sine transform along x and
%   tridiagonal solves along y, and FORM_S uses it when given [25 n].
tests = functiontests(localfunctions);
end

function setupOnce(~)
here = fileparts(mfilename('fullpath'));
addpath(genpath(fileparts(here)));
end

function testStripSolverInvertsTheHalfDomain(testCase)
n = 40;
[A, I1, ~, ~] = poisson_2D(51, n);
A11 = A(I1, I1);
solve = dst_strip_solver(25, n);
rng(3, 'twister');
B = randn(size(A11, 1), 3);
Z = solve(B);
verifyLessThan(testCase, norm(A11 * Z - B, 'fro') / norm(B, 'fro'), 1e-12);
verifyEqual(testCase, size(Z), size(B));
end

function testFastAndCholeskyOperatorsAgree(testCase)
n = 40;
[A, I1, I2, I3] = poisson_2D(51, n);
evalc('mv_chol = form_S(A, I1, I2, I3);');
evalc('mv_fast = form_S(A, I1, I2, I3, [25, n]);');
rng(4, 'twister');
X = randn(n, 5);
Y1 = mv_chol(X);
Y2 = mv_fast(X);
verifyLessThan(testCase, norm(Y1 - Y2, 'fro') / norm(Y1, 'fro'), 1e-12);
end

function testReducedFormIsSelectedAndAgrees(testCase)
n = 40;
[A, I1, I2, I3] = poisson_2D(51, n);
out = evalc('mv_fast = form_S(A, I1, I2, I3, [25, n]);');
verifyTrue(testCase, contains(out, 'Reduced separator form verified'));
S = full(A(I3, I3) - A(I3, I1) * (A(I1, I1) \ A(I1, I3)) - A(I3, I2) * (A(I2, I2) \ A(I2, I3)));
rng(6, 'twister');
X = randn(n, 3);
verifyLessThan(testCase, norm(mv_fast(X) - S * X, 'fro') / norm(S * X, 'fro'), 1e-12);
end

function testWrongDimensionsAreRejected(testCase)
n = 40;
[A, I1, I2, I3] = poisson_2D(51, n);
verifyError(testCase, @() form_S(A, I1, I2, I3, [24, n]), 'form_S:dstDims');
end

function testBuildTestSystemSwitch(testCase)
n = 40;
rng(5, 'twister');
[~, b_fast, ~, ~, mv_fast] = build_test_system(n, 4, 'poisson_schur', 1, struct());
rng(5, 'twister');
[~, b_chol, ~, ~, mv_chol] = build_test_system(n, 4, 'poisson_schur', 1, struct('fast_operator', false));
verifyLessThan(testCase, norm(b_fast - b_chol) / norm(b_chol), 1e-12);
x = (1:n)' / n;
verifyLessThan(testCase, norm(mv_fast(x) - mv_chol(x)) / norm(mv_chol(x)), 1e-12);
end
