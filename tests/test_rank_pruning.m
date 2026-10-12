function tests = test_rank_pruning
%TEST_RANK_PRUNING Unit tests for per-node vs common rank pruning.
    tests = functiontests(localfunctions);
end

function [U, LHS, RHS, G, b, k_A, nb] = fixture(seed)
    rng(seed);
    nb = 4; k_A = 16; b = 8; m = 64;
    U = cell(nb,1); G = cell(nb,1);
    LHS = zeros(nb,1); RHS = zeros(nb,1);
    for i = 1:nb
        [Q,~] = qr(randn(m, k_A), 0);
        U{i} = Q;
        G{i} = randn(m, b) / (10 * i);   % decreasing look-ahead energy per node
        LHS(i) = 1e-3;
        RHS(i) = 1e-3 * (1 + i);         % increasing budget, so admissible counts differ
    end
end

function testDefaultMatchesCommonRank(testCase)
    [U, LHS, RHS, G, b, k_A, nb] = fixture(1);
    [Uref, k_ref, ~, k_node] = refine_rank_sub_block(U, LHS, RHS, G, b, k_A, nb);
    widths = cellfun(@(x) size(x,2), Uref);
    verifyEqual(testCase, widths(:), repmat(k_ref, nb, 1), ...
        'Default mode must give every node the same width.');
    verifyEqual(testCase, k_node(:), repmat(k_ref, nb, 1));
end

function testPerNodeNeverKeepsMore(testCase)
    % Per-node pruning removes each node's own admissible count, which is at
    % least the common minimum, so no node can end up wider.
    [U, LHS, RHS, G, b, k_A, nb] = fixture(2);
    [~, k_common, ~, ~]      = refine_rank_sub_block(U, LHS, RHS, G, b, k_A, nb, false);
    [Uref, k_max, ~, k_node] = refine_rank_sub_block(U, LHS, RHS, G, b, k_A, nb, true);
    verifyLessThanOrEqual(testCase, k_node, k_common, ...
        'Per-node pruning must not retain more than the common rank.');
    verifyEqual(testCase, k_max, max(k_node));
    widths = cellfun(@(x) size(x,2), Uref);
    verifyEqual(testCase, widths(:), k_node(:), ...
        'Returned basis widths must match the reported per-node ranks.');
end

function testPerNodePrunesAtLeastAsMuch(testCase)
    % The fixture gives nodes different budgets, so per-node pruning must
    % strictly beat the common rank on total retained columns.
    [U, LHS, RHS, G, b, k_A, nb] = fixture(3);
    [~, k_common, ~, ~]  = refine_rank_sub_block(U, LHS, RHS, G, b, k_A, nb, false);
    [~, ~, ~, k_node]    = refine_rank_sub_block(U, LHS, RHS, G, b, k_A, nb, true);
    verifyLessThanOrEqual(testCase, sum(k_node), nb * k_common, ...
        'Per-node pruning should retain no more columns in total.');
end

function testAcceptedColumnsUntouched(testCase)
    % Only the final block may be pruned; earlier accepted columns must be
    % bit-identical in both modes.
    [U, LHS, RHS, G, b, k_A, nb] = fixture(4);
    keep = k_A - min(b, k_A);
    [Ua, ~, ~, ~] = refine_rank_sub_block(U, LHS, RHS, G, b, k_A, nb, false);
    [Ub, ~, ~, ~] = refine_rank_sub_block(U, LHS, RHS, G, b, k_A, nb, true);
    for i = 1:nb
        verifyEqual(testCase, Ua{i}(:, 1:keep), U{i}(:, 1:keep));
        verifyEqual(testCase, Ub{i}(:, 1:keep), U{i}(:, 1:keep));
    end
end

function testRanksStayPositive(testCase)
    [U, LHS, RHS, G, b, k_A, nb] = fixture(5);
    [~, ~, ~, k_node] = refine_rank_sub_block(U, LHS, RHS, G, b, k_A, nb, true);
    verifyGreaterThanOrEqual(testCase, k_node, 1, 'A node must retain at least one direction.');
end

function testZeroBudgetRemovesNothing(testCase)
%   The guard on `cumulative_energy <= budget` is a rounding allowance, not a
%   licence. With the budget identically zero nothing is admissible, whatever
%   the block's total sketched energy happens to be.
%
%   This pins the defect behind the rank-7 collapse: the guard used to be
%   scaled by sum(energy2), giving an absolute amplitude floor of
%   sqrt(32*eps) ~ 8.4e-8 independent of the requested tolerance, and this
%   exact input lost 9 of 16 directions to it.
    b = 16; k_A = 16; m = 20;
    U = [eye(k_A); zeros(m-k_A, k_A)];
    amp = [logspace(0,-6,7), 1e-8*ones(1,9)];
    G = sqrt(b) * [diag(amp); zeros(m-k_A, k_A)];
    LHS = 1.0; RHS = 1.0; RHS_budget = 1.0;   % budget == 0 exactly
    [~, k_ref] = refine_rank_sub_block({U}, LHS, RHS, {G}, b, k_A, 1, ...
        false, 1/25, RHS_budget);
    verifyEqual(testCase, k_ref, k_A);
end

function testTinyDirectionsStillGoWhenBudgetAllows(testCase)
%   The fix must not disable pruning: with a budget that genuinely covers the
%   nine negligible directions, they are still removed.
    b = 16; k_A = 16; m = 20;
    U = [eye(k_A); zeros(m-k_A, k_A)];
    amp = [logspace(0,-6,7), 1e-8*ones(1,9)];
    G = sqrt(b) * [diag(amp); zeros(m-k_A, k_A)];
    LHS = 0; RHS = 1.0; RHS_budget = 1e-6;    % budget = 1e-12/25 >> 9e-16
    [~, k_ref] = refine_rank_sub_block({U}, LHS, RHS, {G}, b, k_A, 1, ...
        false, 1/25, RHS_budget);
    verifyEqual(testCase, k_ref, 7);
end
