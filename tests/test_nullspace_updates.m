function tests = test_nullspace_updates
%TEST_NULLSPACE_UPDATES Unit tests for the nullspace constructions.
    tests = functiontests(localfunctions);
end

function testRealUpdates(testCase)
    exercise_updates(testCase, false);
end

function testComplexUpdates(testCase)
    exercise_updates(testCase, true);
end

function testInvalidMethod(testCase)
    Omega0 = eye(4);
    DeltaOmega = randn(4, 2);
    did_error = false;
    try
        initialize_nullspace_basis('unsupported', Omega0, DeltaOmega);
    catch caught
        did_error = contains(caught.message, 'Unknown nullspace method');
    end
    verifyTrue(testCase, did_error);
end

function testMainRejectsInvalidMethod(testCase)
    did_error = false;
    try
        main(struct('method', 'unsupported'));
    catch caught
        did_error = contains(caught.message, 'Unknown nullspace method');
    end
    verifyTrue(testCase, did_error);
end

function testMainRejectsInvalidProbeCount(testCase)
    did_error = false;
    try
        main(struct('operator_probe_count', -1));
    catch caught
        did_error = contains(caught.message, 'operator_probe_count');
    end
    verifyTrue(testCase, did_error);
end

function testMainRejectsInvalidRankPruningScope(testCase)
    did_error = false;
    try
        main(struct('rank_pruning', 'unsupported'));
    catch caught
        did_error = contains(caught.message, 'rank-pruning scope');
    end
    verifyTrue(testCase, did_error);
end

function testSafeguardedFallback(testCase)
    rng(91, 'twister');
    m = 24;
    b = 4;
    [left, ~] = qr(randn(m), 0);
    [right, ~] = qr(randn(m), 0);
    Omega0 = left * diag(logspace(0, -12, m)) * right';
    increments = randn(m, 2*b);
    options = struct('target_tolerance', 1e-9);

    first = increments(:, 1:b);
    Omega = [Omega0, first];
    [Q, state] = initialize_nullspace_basis( ...
        'safeguarded', Omega0, first, options);
    verifyTrue(testCase, state.fallback_active);
    verifyTrue(testCase, state.recomputed_full);
    verifyEqual(testCase, state.full_recompute_count, 1);
    verifyLessThan(testCase, relative_residual(Omega, Q), 1e-12);

    next = increments(:, b+1:2*b);
    Omega = [Omega, next];
    [Q, DeltaQ, state] = extend_nullspace_basis( ...
        'safeguarded', Omega, Q, state, next);
    verifySize(testCase, DeltaQ, [m + 2*b, b]);
    verifyTrue(testCase, state.recomputed_full);
    verifyEqual(testCase, state.full_recompute_count, 2);
    verifyLessThan(testCase, relative_residual(Omega, Q), 1e-12);
end

function testSafeguardedCorrectionImprovesResidual(testCase)
    rng(92, 'twister');
    m = 32;
    b = 4;
    [left, ~] = qr(randn(m), 0);
    [right, ~] = qr(randn(m), 0);
    Omega0 = left * diag(logspace(0, -4, m)) * right';
    DeltaOmega = randn(m, b);
    Omega = [Omega0, DeltaOmega];

    [Q_iterative, ~] = initialize_nullspace_basis( ...
        'iterative', Omega0, DeltaOmega);
    [Q_safeguarded, state] = initialize_nullspace_basis( ...
        'safeguarded', Omega0, DeltaOmega, ...
        struct('target_tolerance', 1e-6));

    iterative_residual = relative_residual(Omega, Q_iterative);
    safeguarded_residual = relative_residual(Omega, Q_safeguarded);
    verifyFalse(testCase, state.fallback_active);
    verifyLessThan(testCase, safeguarded_residual, iterative_residual);
    verifyLessThan(testCase, safeguarded_residual, 1e-14);
end

function testCorrectedUsesResidualInsteadOfConditionEstimate(testCase)
    rng(93, 'twister');
    m = 32;
    b = 4;
    [left, ~] = qr(randn(m), 0);
    [right, ~] = qr(randn(m), 0);
    Omega0 = left * diag(logspace(0, -12, m)) * right';
    DeltaOmega = randn(m, b);
    Omega = [Omega0, DeltaOmega];

    [Q_iterative, ~] = initialize_nullspace_basis( ...
        'iterative', Omega0, DeltaOmega);
    [Q_corrected, state] = initialize_nullspace_basis( ...
        'corrected', Omega0, DeltaOmega, ...
        struct('target_tolerance', 1e-12));

    verifyFalse(testCase, state.recomputed_full);
    verifyEqual(testCase, state.full_recompute_count, 0);
    verifyLessThan(testCase, relative_residual(Omega, Q_corrected), 1e-13);
    verifyLessThan(testCase, relative_residual(Omega, Q_corrected), ...
        relative_residual(Omega, Q_iterative));
end

function testCorrectedFallsBackOnlyAfterFailedCorrection(testCase)
    rng(94, 'twister');
    m = 24;
    b = 4;
    Omega0 = randn(m);
    increments = randn(m, 2*b);
    first = increments(:, 1:b);
    next = increments(:, b+1:end);

    [Q, state] = initialize_nullspace_basis( ...
        'corrected', Omega0, first, struct('target_tolerance', 1e-12));
    state.correction_passes = 0;
    state.U(1, 1) = 1e6 * state.U(1, 1);
    Omega = [Omega0, first, next];
    [Q, ~, state] = extend_nullspace_basis( ...
        'corrected', Omega, Q, state, next);

    verifyTrue(testCase, state.recomputed_full);
    verifyEqual(testCase, state.full_recompute_count, 1);
    verifyLessThan(testCase, relative_residual(Omega, Q), 1e-12);
end

function exercise_updates(testCase, use_complex)
    rng(37, 'twister');
    m = 24;
    b = 4;
    num_batches = 3;

    Omega0 = randn(m);
    increments = randn(m, b * num_batches);
    if use_complex
        Omega0 = (Omega0 + 1i * randn(m)) / sqrt(2);
        increments = (increments + 1i * randn(m, b * num_batches)) / sqrt(2);
    end

    first = increments(:, 1:b);
    Omega = [Omega0, first];
    [Q_naive, state_naive] = initialize_nullspace_basis('naive', Omega0, first);
    [Q_global, state_global] = initialize_nullspace_basis('global', Omega0, first);
    [Q_iterative, state_iterative] = initialize_nullspace_basis('iterative', Omega0, first);
    [Q_safeguarded, state_safeguarded] = initialize_nullspace_basis( ...
        'safeguarded', Omega0, first, struct('target_tolerance', 1e-9));
    [Q_corrected, state_corrected] = initialize_nullspace_basis( ...
        'corrected', Omega0, first, struct('target_tolerance', 1e-9));
    check_methods(testCase, Omega, Q_naive, Q_global, Q_iterative, ...
        Q_safeguarded, Q_corrected, m, b);
    verifyGreaterThan(testCase, state_iterative.initial_rcond, 0);
    verifyGreaterThan(testCase, state_safeguarded.initial_rcond, 0);
    verifyGreaterThan(testCase, state_corrected.initial_rcond, 0);

    for j = 2:num_batches
        next = increments(:, (j-1)*b+1:j*b);
        Omega = [Omega, next];
        [Q_naive, DeltaQ_naive, state_naive] = extend_nullspace_basis( ...
            'naive', Omega, Q_naive, state_naive, next);
        [Q_global, DeltaQ_global, state_global] = extend_nullspace_basis( ...
            'global', Omega, Q_global, state_global, next);
        [Q_iterative, DeltaQ_iterative, state_iterative] = extend_nullspace_basis( ...
            'iterative', Omega, Q_iterative, state_iterative, next);
        [Q_safeguarded, DeltaQ_safeguarded, state_safeguarded] = ...
            extend_nullspace_basis('safeguarded', Omega, ...
            Q_safeguarded, state_safeguarded, next);
        [Q_corrected, DeltaQ_corrected, state_corrected] = ...
            extend_nullspace_basis('corrected', Omega, ...
            Q_corrected, state_corrected, next);

        verifySize(testCase, DeltaQ_naive, [m + j*b, b]);
        verifySize(testCase, DeltaQ_global, [m + j*b, b]);
        verifySize(testCase, DeltaQ_iterative, [m + j*b, b]);
        verifySize(testCase, DeltaQ_safeguarded, [m + j*b, b]);
        verifySize(testCase, DeltaQ_corrected, [m + j*b, b]);
        check_methods(testCase, Omega, Q_naive, Q_global, Q_iterative, ...
            Q_safeguarded, Q_corrected, m, j*b);
    end
end

function check_methods(testCase, Omega, Q_naive, Q_global, Q_iterative, ...
        Q_safeguarded, Q_corrected, m, rank)
    verifySize(testCase, Q_naive, [m + rank, rank]);
    verifySize(testCase, Q_global, [m + rank, rank]);
    verifySize(testCase, Q_iterative, [m + rank, rank]);
    verifySize(testCase, Q_safeguarded, [m + rank, rank]);
    verifySize(testCase, Q_corrected, [m + rank, rank]);

    scale = max(norm(Omega, 'fro'), realmin);
    verifyLessThan(testCase, norm(Omega * Q_naive, 'fro') / scale, 1e-11);
    verifyLessThan(testCase, norm(Omega * Q_global, 'fro') / scale, 1e-11);
    verifyLessThan(testCase, norm(Omega * Q_iterative, 'fro') / scale, 1e-11);
    verifyLessThan(testCase, norm(Omega * Q_safeguarded, 'fro') / scale, 1e-11);
    verifyLessThan(testCase, norm(Omega * Q_corrected, 'fro') / scale, 1e-11);
    verifyLessThan(testCase, ...
        norm(Q_naive' * Q_naive - eye(rank, 'like', Q_naive), 'fro'), 1e-11);
    verifyLessThan(testCase, ...
        norm(Q_global' * Q_global - eye(rank, 'like', Q_global), 'fro'), 1e-11);
    verifyLessThan(testCase, ...
        norm(Q_iterative' * Q_iterative - eye(rank, 'like', Q_iterative), 'fro'), 1e-11);
    verifyLessThan(testCase, norm(Q_safeguarded' * Q_safeguarded - ...
        eye(rank, 'like', Q_safeguarded), 'fro'), 1e-11);
    verifyLessThan(testCase, norm(Q_corrected' * Q_corrected - ...
        eye(rank, 'like', Q_corrected), 'fro'), 1e-11);

    projector_naive_global = norm( ...
        Q_naive * Q_naive' - Q_global * Q_global', 'fro');
    projector_naive_iterative = norm( ...
        Q_naive * Q_naive' - Q_iterative * Q_iterative', 'fro');
    projector_naive_safeguarded = norm( ...
        Q_naive * Q_naive' - Q_safeguarded * Q_safeguarded', 'fro');
    projector_naive_corrected = norm( ...
        Q_naive * Q_naive' - Q_corrected * Q_corrected', 'fro');
    verifyLessThan(testCase, projector_naive_global, 1e-10);
    verifyLessThan(testCase, projector_naive_iterative, 1e-10);
    verifyLessThan(testCase, projector_naive_safeguarded, 1e-10);
    verifyLessThan(testCase, projector_naive_corrected, 1e-10);
end

function value = relative_residual(Omega, Q)
    value = norm(Omega * Q, 'fro') / max(norm(Omega, 'fro'), realmin);
end
