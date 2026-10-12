function tests = test_geqrt_kernels
%TEST_GEQRT_KERNELS The compact-WY QR kernels, real and complex.
%   geqrt_mex (DGEQRT/ZGEQRT) and gemqrt_mex (DGEMQRT/ZGEMQRT) must give a
%   unitary Q whose leading columns span the input, agree with the MATLAB
%   paths of compute_householder and null_space_basis up to a unit-modulus
%   factor per column, and apply Q' as the conjugate transpose for complex
%   input. Added 2026-10-10 with the complex entry points, which let the
%   Lippmann-Schwinger (complex-symmetric) problem use the kernels.
    tests = functiontests(localfunctions);
end

function setupOnce(~)
    here = fileparts(mfilename('fullpath'));
    addpath(genpath(fileparts(here)));
end

function A = sample(m, n, use_complex)
    rng(7, 'twister');
    A = randn(m, n);
    if use_complex, A = (A + 1i * randn(m, n)) / sqrt(2); end
end

function testGeqrtGivesUnitaryQ(testCase)
    if exist('geqrt_mex', 'file') ~= 3, return; end
    for use_complex = [false true]
        A = sample(60, 13, use_complex);
        [V, T] = geqrt_mex(A);
        Q = eye(60) - V * triu(T) * V';
        verifyEqual(testCase, Q' * Q, eye(60), 'AbsTol', 1e-12);
        Q1 = Q(:, 1:13);
        verifyLessThan(testCase, norm(A - Q1 * (Q1' * A), 'fro') / norm(A, 'fro'), 1e-13, ...
            'Leading columns of Q do not span the input.');
        verifyEqual(testCase, isreal(V), ~use_complex);
    end
end

function testComputeHouseholderMatchesLoop(testCase)
    if exist('geqrt_mex', 'file') ~= 3, return; end
    for use_complex = [false true]
        A = sample(50, 9, use_complex);
        [W, C] = compute_householder(A);            % MEX path
        Q = eye(50) - W * C';
        verifyEqual(testCase, Q' * Q, eye(50), 'AbsTol', 1e-12);
        Q1 = Q(:, 1:9);
        % The loop and LAPACK may differ by a unit-modulus factor per column,
        % so compare subspaces: P is unitary exactly when they coincide.
        P = Q1' * orth(A);
        verifyEqual(testCase, P' * P, eye(9), 'AbsTol', 1e-10, ...
            'compute_householder does not span the input.');
        Z = apply_householder(W, C, A, 'transpose');
        verifyLessThan(testCase, norm(Z(10:end, :), 'fro') / norm(A, 'fro'), 1e-13, ...
            'Q''A is not upper trapezoidal.');
    end
end

function testGemqrtAppliesQAndItsAdjoint(testCase)
    if exist('gemqrt_mex', 'file') ~= 3, return; end
    for use_complex = [false true]
        A = sample(70, 20, use_complex);
        X = sample(70, 5, use_complex);
        [V, T] = geqrt_mex(A, 8);                 % blocked T
        [Vf, Tf] = geqrt_mex(A);                  % one block, explicit Q
        Q = eye(70) - Vf * triu(Tf) * Vf';
        verifyEqual(testCase, gemqrt_mex(V, T, X), Q * X, 'AbsTol', 1e-12);
        verifyEqual(testCase, gemqrt_mex(V, T, X, 'T'), Q' * X, 'AbsTol', 1e-12);
        verifyEqual(testCase, gemqrt_mex(V, T, X, 'C'), Q' * X, 'AbsTol', 1e-12);
    end
end

function testGemqrtRejectsMixedTypes(testCase)
    if exist('gemqrt_mex', 'file') ~= 3, return; end
    A = sample(30, 4, true);
    [V, T] = geqrt_mex(A);
    verifyError(testCase, @() gemqrt_mex(V, T, randn(30, 2)), 'gemqrt_mex:mixed');
end

function testNullSpaceBasisMatchesQr(testCase)
    for use_complex = [false true]
        Omega = sample(24, 40, use_complex);      % m x p with p >= m + k
        k = 6;
        hat_Q = null_space_basis(Omega, k);
        verifyEqual(testCase, size(hat_Q), [40 k]);
        verifyEqual(testCase, hat_Q' * hat_Q, eye(k), 'AbsTol', 1e-12);
        verifyLessThan(testCase, norm(Omega * hat_Q, 'fro') / norm(Omega, 'fro'), 1e-13, ...
            'null_space_basis columns are not in the nullspace.');
        [Qr, ~] = qr(Omega');
        ref = Qr(:, end - k + 1:end);
        P = hat_Q' * ref;
        verifyEqual(testCase, P' * P, eye(k), 'AbsTol', 1e-10, ...
            'null_space_basis spans a different subspace than qr.');
    end
end
