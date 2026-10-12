function [U_refined_all, k_refined, retained_last_all, k_per_node] = ...
        refine_rank_sub_block(U_b_i_all, LHS_all, RHS_all, A_12_Psi_all, ...
        b, k_A, num_blocks, per_node, budget_fraction, RHS_budget_all)
%REFINE_RANK_SUB_BLOCK Prune low-energy directions from the last basis block.
%   The look-ahead sample provides a sketched energy for each column of the
%   final accepted basis block. Each node ranks those energies separately.
%
%   With PER_NODE false (default) the routine removes the largest number of
%   directions admissible at *every* node, which keeps one rank per level for
%   the rest of the factorization. That is throttled by the worst node: with
%   eight leaf nodes the median node typically admits twice the removals that
%   are actually applied.
%
%   With PER_NODE true each node removes its own admissible count, so the
%   returned bases have different widths. K_PER_NODE reports the retained
%   rank at each node in both modes, and K_REFINED is the largest of them,
%   which callers still use to size levelwise storage.

    if nargin < 8 || isempty(per_node), per_node = false; end
    % Callers should pass the fraction they intend; building_bases passes
    % 1/L^2 so that pruning spends at most eps/L of the norm budget per
    % level. The default of 1 spends the whole local slack and exists only
    % so direct callers and tests need not supply it.
    if nargin < 9 || isempty(budget_fraction), budget_fraction = 1; end
    % Threshold the pruning budget is measured against. Defaults to the
    % stopping threshold; callers pass the unrelaxed one so that the
    % level-dependent relaxation does not also license more pruning.
    if nargin < 10 || isempty(RHS_budget_all), RHS_budget_all = RHS_all; end
    if numel(U_b_i_all) ~= num_blocks || ...
            numel(A_12_Psi_all) ~= num_blocks
        error('Rank-pruning inputs must contain one entry per block.');
    end

    last_width = min(b, k_A);
    max_removable = min(last_width, max(k_A - 1, 0));
    removal_order_all = cell(num_blocks, 1);
    removable_count = zeros(num_blocks, 1);

    for block = 1:num_blocks
        U = U_b_i_all{block};
        if size(U, 2) < k_A
            error('Basis at block %d has fewer than k_A columns.', block);
        end

        U_last = U(:, k_A-last_width+1:k_A);
        G_lookahead = A_12_Psi_all{block};
        T = (U_last' * G_lookahead) / sqrt(b);
        energy2 = sum(abs(T).^2, 2);
        [~, order] = sort(energy2, 'ascend');
        removal_order_all{block} = order;

        % Spend only a fraction of the local slack. Pruning at one level
        % raises the error entering the next, and the per-level stopping test
        % does not account for that accumulation, so consuming the whole
        % budget at every level compounds up the tree.
        budget = budget_fraction * ...
            max(RHS_budget_all(block)^2 - LHS_all(block)^2, 0);
        cumulative_energy = cumsum(energy2(order));
        % The slack is a rounding guard on `cumulative_energy <= budget`, so
        % it is scaled by the budget, which is the quantity it is compared
        % against. Scaling it by the block's total sketched energy instead
        % turned it into an absolute floor: sum(energy2) is O(||A_12||^2)
        % while the budget is O(tol^2 ||A_12||^2), so at tol = 1e-12 the
        % guard exceeded what it guarded by eleven orders and the test
        % degenerated into "drop every direction below 32*eps of the block
        % energy" -- an amplitude floor of sqrt(32*eps) ~ 8.4e-8 whatever
        % tolerance was requested. When the budget is zero nothing is
        % admissible and the guard has to say so.
        comparison_slack = 32 * eps(max(budget, realmin));
        removable_count(block) = min( ...
            sum(cumulative_energy <= budget + comparison_slack), ...
            max_removable);
    end

    common_remove = min(removable_count);

    % How much pruning the levelwise common rank costs: each node's own
    % admissible removal against the common value actually applied.
    record_pruning_diagnostic(removable_count, common_remove, k_A, last_width);

    if per_node
        applied_remove = removable_count;
    else
        applied_remove = repmat(common_remove, num_blocks, 1);
    end

    k_per_node = k_A - applied_remove(:);
    k_refined = max(k_per_node);
    U_refined_all = cell(num_blocks, 1);
    retained_last_all = cell(num_blocks, 1);

    for block = 1:num_blocks
        U = U_b_i_all{block}(:, 1:k_A);
        U_previous = U(:, 1:k_A-last_width);
        U_last = U(:, k_A-last_width+1:k_A);

        remove = removal_order_all{block}(1:applied_remove(block));
        retain = setdiff((1:last_width)', remove, 'stable');
        U_refined_all{block} = [U_previous, U_last(:, retain)];
        retained_last_all{block} = retain;
    end
end

function record_pruning_diagnostic(removable_count, common_remove, k_A, last_width)
%RECORD_PRUNING_DIAGNOSTIC Log per-node admissible removals against the
%   common removal imposed by the levelwise rank. Inert unless enabled.
    global PRUNE_DIAG %#ok<GVMIS>
    if isempty(PRUNE_DIAG) || ~isstruct(PRUNE_DIAG) || ~PRUNE_DIAG.enabled
        return;
    end
    PRUNE_DIAG.rows(end+1, :) = [numel(removable_count), k_A, last_width, ...
        common_remove, min(removable_count), median(removable_count), ...
        max(removable_count), sum(removable_count)];
end

