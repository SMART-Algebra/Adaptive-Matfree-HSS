function raw = merge_block_size_shards(output_dir, shard_dirs)
%MERGE_BLOCK_SIZE_SHARDS Join block-size sweeps run as separate array tasks.
%   RAW = MERGE_BLOCK_SIZE_SHARDS(OUTPUT_DIR, SHARD_DIRS) reads
%   block_size_raw.csv from each directory in the cell array SHARD_DIRS,
%   concatenates the rows, and writes block_size_raw.csv,
%   block_size_summary.csv and block_size_study.mat into OUTPUT_DIR in the
%   layout run_block_size_study produces, so the table builders read a
%   merged sweep and a single-task sweep the same way. Rows must come from
%   one code commit; mixing commits is refused.
    if ischar(shard_dirs) || isstring(shard_dirs), shard_dirs = cellstr(string(shard_dirs)); end
    T = table();
    for q = 1:numel(shard_dirs)
        f = fullfile(shard_dirs{q}, 'block_size_raw.csv');
        if ~isfile(f), error('merge_block_size_shards:missing', 'No block_size_raw.csv in %s.', shard_dirs{q}); end
        T = [T; read_results_table(f)]; %#ok<AGROW>
    end
    if isempty(T), error('merge_block_size_shards:empty', 'No rows to merge.'); end
    if ismember('code_commit', T.Properties.VariableNames)
        commits = unique(T.code_commit);
        if numel(commits) > 1
            error('merge_block_size_shards:commits', ...
                'Shards come from different code commits (%s); rerun them together.', strjoin(commits, ', '));
        end
    end
    key = strcat(T.case_name, "|", string(T.step_b), "|", T.rank_pruning, "|", string(T.seed));
    if numel(unique(key)) ~= numel(key)
        error('merge_block_size_shards:duplicate', 'The shards overlap: the same (case, b, scope, seed) appears twice.');
    end
    if ~isfolder(output_dir), mkdir(output_dir); end
    raw = table2struct(T)';
    writetable(T, fullfile(output_dir, 'block_size_raw.csv'));
    write_block_size_summary(raw, output_dir);
    code_commit = code_commit_hash(); %#ok<NASGU>
    save(fullfile(output_dir, 'block_size_study.mat'), 'raw', 'code_commit');
    fprintf('Merged %d runs from %d shards into %s\n', height(T), numel(shard_dirs), output_dir);
end
