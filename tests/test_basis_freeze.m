function tests = test_basis_freeze
%TEST_BASIS_FREEZE A node that passes the stopping test stops growing.
    tests = functiontests(localfunctions);
end

function testConvergedNodesDoNotSaturateTheLeaf(testCase)
    % Regression test for the basis-growth breakdown found on 2026-09-11.
    % The level loop used to append b columns to every node on every batch
    % until the slowest node passed. A node that had passed early then sat
    % at the roundoff floor, the appended columns were noise, a single
    % reorthogonalization could not keep U orthonormal, and the stopping
    % estimate exploded, so the level ran to full rank. On this
    % configuration every method except naive returned the leaf at rank
    % 256 while naive certified at 240. With converged nodes frozen, the
    % row update certifies the leaf near 237 like naive.
    root_dir = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(root_dir));
    params = struct('dim', 2, 'kernel_type', 'matern', ...
        'kernel_params', struct('nu', 1.5, 'ell', 0.1), ...
        'user_tolerance', 1e-12, 'step_b', 8, 'num_levels', 3, ...
        'pt_leaf', 256, 'method', 'householder', 'rank_pruning', 'all', ...
        'sample_distribution', 'auto', 'compute_cond', false, ...
        'warmup', false, 'collect_diagnostics', true, ...
        'operator_probe_count', 0, 'unpreconditioned_baseline', false, ...
        'comparison_seed', 0);
    rng(0, 'twister');
    [captured, ~, stats] = evalc('main(params)');
    verifyFalse(testCase, contains(captured, 'Rank search reached'), ...
        'The leaf level ran to full rank with the stopping test failing.');
    leaf_rank = leaf_rank_from(stats.level_ranks, params.num_levels);
    verifyLessThanOrEqual(testCase, leaf_rank, ...
        params.pt_leaf - params.step_b, ...
        'The leaf basis was not certified below the node dimension.');
    verifyGreaterThan(testCase, leaf_rank, 200, ...
        'The leaf rank is implausibly small for this kernel at 1e-12.');
    verifyLessThan(testCase, stats.solve_residual, 1e-10);
end

function k = leaf_rank_from(text, leaf_level)
%LEAF_RANK_FROM Parse "level:dim:rank|..." for the given level.
    tokens = regexp(char(text), ...
        sprintf('(?<![0-9])%d:(\\d+):(\\d+)', leaf_level), 'tokens', 'once');
    assert(~isempty(tokens), 'Level %d not found in "%s".', ...
        leaf_level, char(text));
    k = str2double(tokens{2});
end
