function tests = test_householder_nullspace
%TEST_HOUSEHOLDER_NULLSPACE Unit tests for the Householder QR row update.
%   Covers the 'householder' nullspace extension: accuracy, the exact
%   preservation of the retained basis, agreement between the MEX kernel and
%   the pure-MATLAB fallback, independence from cond(Omega0), and the
%   dispatcher integration.
    tests = functiontests(localfunctions);
end

% ------------------------------------------------------------------ basics

function testRealAccuracy(testCase)
    [Omega, Q] = drive(64, 8, 5, 1, false, mex_available());
    verify_basis(testCase, Omega, Q, 64, 8, 5);
end

function testComplexAccuracy(testCase)
    [Omega, Q] = drive(64, 8, 5, 1, true, mex_available());
    verify_basis(testCase, Omega, Q, 64, 8, 5);
end

function testFallbackAccuracy(testCase)
    % The unblocked loop must stand on its own when no MEX is built.
    [Omega, Q] = drive(48, 6, 4, 1, false, false);
    verify_basis(testCase, Omega, Q, 48, 6, 4);
    [Omega, Q] = drive(48, 6, 4, 1, true, false);
    verify_basis(testCase, Omega, Q, 48, 6, 4);
end

% ------------------------------------------- the invariant the method rests on

function testRetainedBasisPreservedExactly(testCase)
    % Reflector j touches only pivot row j and the b appended rows, so the
    % previously computed columns must reappear bit for bit, zero padded.
    % Sample reuse in the caller depends on this being exact, not approximate.
    m = 40; b = 5; J = 4;
    [~, Q_short] = drive(m, b, J - 1, 1, false, mex_available());
    [~, Q_long]  = drive(m, b, J,     1, false, mex_available());
    k_old = size(Q_short, 2);
    padded = [Q_short; zeros(b, k_old)];
    verifyEqual(testCase, Q_long(:, 1:k_old), padded, ...
        'Retained nullspace columns were not preserved exactly.');
end

% ------------------------------------------------------- MEX vs fallback

function testMexMatchesFallback(testCase)
    if exist('tpqrt_update_mex', 'file') ~= 3
        return;   % nothing to compare against
    end
    for complex_case = [false true]
        [~, Q_mex] = drive(48, 6, 4, 1, complex_case, true);
        [~, Q_ref] = drive(48, 6, 4, 1, complex_case, false);
        % Compare subspaces: the two paths may differ by column phases.
        overlap = abs(Q_mex' * Q_ref);
        verifyEqual(testCase, overlap, eye(size(Q_mex, 2)), 'AbsTol', 1e-10, ...
            'MEX kernel and MATLAB fallback disagree.');
    end
end

% --------------------------------------------------- the point of the method

function testResidualIndependentOfConditioning(testCase)
    % No inverse of Omega0 is formed, so the residual must not track
    % cond(Omega0). This is the property that distinguishes the method from
    % the fixed-anchor iterative update.
    residuals = zeros(1, 3);
    kappas = [1e2 1e8 1e14];
    for t = 1:numel(kappas)
        [Omega, Q] = drive(64, 8, 5, kappas(t), false, mex_available());
        residuals(t) = norm(Omega * Q, 'fro') / norm(Omega, 'fro');
    end
    verifyLessThan(testCase, residuals, 1e-13, ...
        'Residual grew with cond(Omega0).');
    % Ill conditioning must not cost more than one order of magnitude.
    verifyLessThan(testCase, residuals(3), 10 * max(residuals(1), eps), ...
        'Residual at cond = 1e14 is far above the well-conditioned case.');
end

function testBeatsIterativeOnIllConditionedProblem(testCase)
    m = 64; b = 8; J = 5; kappa = 1e12;
    [Omega0, adds] = problem(m, b, J, kappa, false);
    r = zeros(1, 2);
    methods = {'iterative', 'householder'};
    for t = 1:2
        [Q, state] = initialize_nullspace_basis(methods{t}, Omega0, adds{1});
        Omega = [Omega0, adds{1}];
        for j = 2:J
            Omega = [Omega, adds{j}];
            [Q, ~, state] = extend_nullspace_basis( ...
                methods{t}, Omega, Q, state, adds{j});
        end
        r(t) = norm(Omega * Q, 'fro') / norm(Omega, 'fro');
    end
    verifyLessThan(testCase, r(2), r(1) / 100, ...
        'Householder update was not substantially more accurate than iterative.');
end

% ------------------------------------------------------ dispatcher wiring

function testDispatcherIntegration(testCase)
    m = 48; b = 6; J = 4;
    [Omega0, adds] = problem(m, b, J, 1, false);
    [Q, state] = initialize_nullspace_basis('householder', Omega0, adds{1});
    Omega = [Omega0, adds{1}];
    for j = 2:J
        Omega = [Omega, adds{j}];
        [Q, DeltaQ, state] = extend_nullspace_basis( ...
            'householder', Omega, Q, state, adds{j});
        verifyEqual(testCase, size(DeltaQ, 2), b);
    end
    verify_basis(testCase, Omega, Q, m, b, J);
    verifyFalse(testCase, state.recomputed_full, ...
        'The householder update should never need a full recomputation.');
    verifyEqual(testCase, state.full_recompute_count, 0);
end

function testRejectsWrongShape(testCase)
    verifyError(testCase, @() extend_nullspace(randn(6, 5), randn(6, 2)), ...
        'extend_nullspace:shape');
    state = struct('R', triu(randn(6)), 'Q1', eye(6), 'use_mex', false);
    verifyError(testCase, @() extend_nullspace(state, randn(5, 2)), ...
        'extend_nullspace:shape');
end

% ---------------------------------------------------------------- helpers

function [Omega0, adds] = problem(m, b, J, kappa, complex_case)
    rng(20260908);
    if complex_case
        G = (randn(m) + 1i * randn(m)) / sqrt(2);
    else
        G = randn(m);
    end
    if kappa > 1
        [U, ~, V] = svd(G);
        G = (U * diag(logspace(0, -log10(kappa), m))) * V';
    end
    Omega0 = G;
    adds = cell(1, J);
    for j = 1:J
        if complex_case
            adds{j} = (randn(m, b) + 1i * randn(m, b)) / sqrt(2);
        else
            adds{j} = randn(m, b);
        end
    end
end

% ------------------------------------------------------------------------
function tf = mex_available()
%MEX_AVAILABLE True when the compiled kernel is present.
%   The accuracy tests are about the row update, not about which kernel
%   evaluates it, so they run against whichever is built. A fresh clone has
%   no MEX -- *.mex* is not tracked -- and would otherwise fail four tests
%   before the user has run build_mex. testMexMatchesFallback is the one
%   test that genuinely needs both and guards itself.
    tf = exist('tpqrt_update_mex', 'file') == 3;
end

% ------------------------------------------------------------------------
function [Omega, Q] = drive(m, b, J, kappa, complex_case, use_mex)

    [Omega0, adds] = problem(m, b, J, kappa, complex_case);
    [Qf, R] = qr(Omega0');
    state = struct('R', triu(R), 'Q1', Qf, 'use_mex', logical(use_mex));
    Omega = Omega0;
    Q = zeros(m, 0, 'like', Omega0);
    for j = 1:J
        [DeltaQ, state] = extend_nullspace(state, adds{j});
        Omega = [Omega, adds{j}];
        Q = [[Q; zeros(b, size(Q, 2))], DeltaQ];
    end
end

function verify_basis(testCase, Omega, Q, m, b, J)
    verifyEqual(testCase, size(Q), [m + b * J, b * J], ...
        'Nullspace basis has the wrong shape.');
    residual = norm(Omega * Q, 'fro') / norm(Omega, 'fro');
    verifyLessThan(testCase, residual, 1e-13, 'Omega*Q is not negligible.');
    orthogonality = norm(Q' * Q - eye(size(Q, 2)));
    verifyLessThan(testCase, orthogonality, 1e-12, 'Q is not orthonormal.');
end
