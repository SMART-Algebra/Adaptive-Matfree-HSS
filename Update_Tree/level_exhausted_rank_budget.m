function tf = level_exhausted_rank_budget(still_failing, accepted_width, block_size, step_b)
%LEVEL_EXHAUSTED_RANK_BUDGET Did any node actually run out of rank?
%   TF = LEVEL_EXHAUSTED_RANK_BUDGET(STILL_FAILING, ACCEPTED_WIDTH,
%   BLOCK_SIZE, STEP_B) is true when some node both fails its stopping test
%   and has an accepted basis at or beyond the widest certifiable width,
%   BLOCK_SIZE - STEP_B. The stopping test holds out the most recent block
%   of STEP_B samples, so no wider basis can ever be certified.
%
%   This separates the two reasons the rank search can stop with a node
%   still failing:
%
%     * the node has exhausted its rank budget, which means the leaf is too
%       small for the kernel at this tolerance and the level genuinely has
%       to be stored at full rank;
%     * the level merely ran out of its sample budget while a node was
%       marginal. The stopping test is a random estimate with a relative
%       standard error of about 1/sqrt(2*STEP_B), so a node whose true error
%       sits near the tolerance fails a fraction of its re-tests however
%       many samples it is given. Storing the level at full rank because of
%       one such node discards every accepted basis at that level.
%
%   Treating the second case as the first was a bug: on 1D Poisson at
%   N = 262144, eps = 1e-9, b = 16, seed 0, exactly one leaf of 1024 was
%   marginal (basis of 16 columns, LHS/RHS between 0.41 and 0.85) and its
%   estimate came in at 1.03 on the final trial. The level was stored at
%   rank 256 rather than 32, costing 1189 MB against 613 and 672 operator
%   applications against 304.
%
%   See also BUILDING_BASES.
    ceiling = max(block_size - step_b, 0);
    tf = any(still_failing(:) & (accepted_width(:) >= ceiling));
end
