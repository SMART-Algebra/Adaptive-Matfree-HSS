function tests = test_common_rank_widest_node
%TEST_COMMON_RANK_WIDEST_NODE The level's common rank is the widest node, not trials times b.
%   Regression for the 2026-09-12 finding: on 1D Poisson at eps = 1e-9 with
%   b = 16 the leaf level of the seed-2 run took five trials because
%   marginal frozen nodes failed re-tests in turn, and the common rank was
%   set to 5 * 16 = 80 (64 after pruning) although no node had accepted
%   more than 32 columns. With the common rank equal to the widest accepted
%   basis the leaf comes out at 18 or 19, as it does for every other seed.
tests = functiontests(localfunctions);
end

function setupOnce(~)
here = fileparts(mfilename('fullpath'));
addpath(genpath(fileparts(here)));
end

function testLeafRankIsBoundedByWidestNode(testCase)
params = struct('dim', 1, 'kernel_type', 'poisson_schur', 'kernel_params', struct(), ...
    'user_tolerance', 1e-9, 'step_b', 16, 'num_levels', 7, 'pt_leaf', 256, ...
    'method', 'householder', 'rank_pruning', 'all', 'warmup', false, ...
    'compute_cond', false, 'comparison_seed', 2);
evalc('[~, stats] = main(params);');
leaf = stats.level_stats(1);
verifyEqual(testCase, leaf.level, 7);
% No node of this problem needs more than two blocks of 16 at this
% tolerance; trials times b would have given 80 here.
verifyLessThanOrEqual(testCase, leaf.rank_before_pruning, 32, ...
    'Leaf rank before pruning exceeds the widest node.');
verifyLessThanOrEqual(testCase, leaf.rank, 24, 'Leaf rank inflated.');
verifyLessThan(testCase, stats.memory_mb, 40, 'Storage inflated by the stalled leaf level.');
verifyLessThan(testCase, stats.solve_residual, 1e-8);
end
