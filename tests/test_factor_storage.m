function tests = test_factor_storage
%TEST_FACTOR_STORAGE Verify construction samples are not reported as factors.
    tests = functiontests(localfunctions);
end

function testDiscardSamplingState(testCase)
    node = representative_nodes();
    node = discard_sampling_state(node);

    sample_fields = { ...
        'Y_tilde', 'Omega_tilde', 'Y_tilde_n', 'Omega_tilde_n'};
    for node_id = 1:numel(node)
        for field_id = 1:numel(sample_fields)
            verifyEmpty(testCase, node(node_id).(sample_fields{field_id}));
        end
    end
    verifyNotEmpty(testCase, node(1).U.W);
    verifyNotEmpty(testCase, node(1).X_rr);
end

function testSamplesDoNotChangeReportedStorage(testCase)
    node = representative_nodes();
    with_samples = memory_stats(node);
    node = discard_sampling_state(node);
    without_samples = memory_stats(node);

    verifyEqual(testCase, with_samples, without_samples);
end

function node = representative_nodes()
    node = repmat(struct( ...
        'U', struct('W', randn(12, 3), 'C', randn(12, 3)), ...
        'X_rr', randn(9), ...
        'Y_tilde', randn(12, 20), ...
        'Omega_tilde', randn(12, 20), ...
        'Y_tilde_n', complex(randn(12, 20), randn(12, 20)), ...
        'Omega_tilde_n', complex(randn(12, 20), randn(12, 20)), ...
        'isLeaf', true, ...
        'rank', 3), 2, 1);
end
