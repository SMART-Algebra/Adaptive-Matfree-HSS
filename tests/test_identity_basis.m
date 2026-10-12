function tests = test_identity_basis
%TEST_IDENTITY_BASIS A node whose rank search reached the node dimension is
%   passed to its parent untransformed, and costs no storage for U.
    tests = functiontests(localfunctions);
end

function setupOnce(testCase) %#ok<*INUSD>
    here = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(here));
end

function testSquareBasisBecomesIdentity(testCase)
    U = orth(randn(12));
    s = wy_or_identity(U);
    verifyEqual(testCase, size(s.W), [12 0]);
    verifyEqual(testCase, size(s.C), [12 0]);
end

function testTallBasisKeepsItsWYPair(testCase)
    U = orth(randn(12, 5));
    s = wy_or_identity(U);
    verifyEqual(testCase, size(s.W, 2), 5);
    verifyEqual(testCase, size(s.C, 2), 5);
end

function testEmptyPairAppliesAsIdentity(testCase)
    % apply_householder forms Q = I - W*C'. With W and C of size m-by-0 that
    % product is the m-by-m zero matrix, so every call site is a no-op
    % without needing a branch of its own. This is the property the whole
    % change rests on.
    s = wy_or_identity(orth(randn(9)));
    X = randn(9, 4);
    verifyEqual(testCase, apply_householder(s.W, s.C, X, 'transpose'), X);
    verifyEqual(testCase, apply_householder(s.W, s.C, X, 'normal'), X);
end

function testComplexSquareBasisBecomesIdentity(testCase)
    U = orth(randn(8) + 1i*randn(8));
    s = wy_or_identity(U);
    verifyEqual(testCase, size(s.W), [8 0]);
    verifyTrue(testCase, ~isreal(s.W) || isempty(s.W));
    X = randn(8, 3) + 1i*randn(8, 3);
    verifyEqual(testCase, apply_householder(s.W, s.C, X, 'transpose'), X);
end
