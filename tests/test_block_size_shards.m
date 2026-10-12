function tests = test_block_size_shards
%TEST_BLOCK_SIZE_SHARDS Sharded block-size sweeps merge into the single-task layout.
tests = functiontests(localfunctions);
end

function setupOnce(~)
here = fileparts(mfilename('fullpath'));
addpath(genpath(fileparts(here)));
end

function T = shard_rows(scope, seeds, commit)
n = numel(seeds);
T = table(repmat("poisson_2d", n, 1), repmat(4, n, 1), repmat(string(scope), n, 1), seeds(:), ...
    repmat(16384, n, 1), repmat(1e-9, n, 1), repmat(256, n, 1), repmat(4, n, 1), ...
    500 + seeds(:), repmat("4:256:200", n, 1), repmat(1000, n, 1), 100 + seeds(:), ...
    repmat(20, n, 1), repmat(120, n, 1), 400 + seeds(:), repmat(1e-8, n, 1), repmat(130, n, 1), ...
    repmat(string(commit), n, 1), 'VariableNames', {'case_name', 'step_b', 'rank_pruning', 'seed', ...
    'n', 'tolerance', 'pt_leaf', 'num_levels', 'final_rank', 'level_ranks', 'n_matvec', 't_factor', ...
    'total_matvec_time', 'total_build_time', 'memory_mb', 'solve_residual', 'wall_time', 'code_commit'});
end

function d = write_shard(root, name, T)
d = fullfile(root, name); mkdir(d);
writetable(T, fullfile(d, 'block_size_raw.csv'));
end

function testMergeProducesSummaryWithOneRowPerScope(testCase)
root = tempname; mkdir(root); cleanup = onCleanup(@() rmdir(root, 's')); %#ok<NASGU>
c = code_commit_hash();
d1 = write_shard(root, 'all', shard_rows('all', [0 1 2 3], c));
d2 = write_shard(root, 'leaf', shard_rows('leaf', [0 1 2 3], c));
d3 = write_shard(root, 'none', shard_rows('none', [0 1 2 3], c));
out = fullfile(root, 'merged');
raw = merge_block_size_shards(out, {d1, d2, d3});
verifyEqual(testCase, numel(raw), 12);
S = read_results_table(fullfile(out, 'block_size_summary.csv'));
verifyEqual(testCase, height(S), 3);
verifyEqual(testCase, sort(S.rank_pruning), sort(["all"; "leaf"; "none"]));
verifyEqual(testCase, S.n_seeds, [4; 4; 4]);
verifyEqual(testCase, S.memory_mb_med, [401.5; 401.5; 401.5]);
verifyTrue(testCase, isfile(fullfile(out, 'block_size_raw.csv')));
verifyTrue(testCase, isfile(fullfile(out, 'block_size_study.mat')));
end

function testMergeRefusesOverlapAndMixedCommits(testCase)
root = tempname; mkdir(root); cleanup = onCleanup(@() rmdir(root, 's')); %#ok<NASGU>
c = code_commit_hash();
d1 = write_shard(root, 'a', shard_rows('all', [0 1], c));
d2 = write_shard(root, 'b', shard_rows('all', [1 2], c));
verifyError(testCase, @() merge_block_size_shards(fullfile(root, 'm1'), {d1, d2}), ...
    'merge_block_size_shards:duplicate');
d3 = write_shard(root, 'c', shard_rows('leaf', [0 1], 'deadbee'));
verifyError(testCase, @() merge_block_size_shards(fullfile(root, 'm2'), {d1, d3}), ...
    'merge_block_size_shards:commits');
end

function testScopeArgumentRestrictsTheSweep(testCase)
out = tempname; mkdir(out); cleanup = onCleanup(@() rmdir(out, 's')); %#ok<NASGU>
evalc('run_block_size_study(out, ''poisson_1d'', 3, {''leaf''});');
T = read_results_table(fullfile(out, 'block_size_raw.csv'));
verifyEqual(testCase, unique(T.rank_pruning), "leaf");
verifyEqual(testCase, height(T), 24);
verifyError(testCase, @() run_block_size_study(out, 'poisson_1d', 3, {'bogus'}), ...
    'run_block_size_study:scope');
end
