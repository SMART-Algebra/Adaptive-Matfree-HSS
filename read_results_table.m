function T = read_results_table(file)
%READ_RESULTS_TABLE Read a results CSV with provenance columns kept as text.
%   T = READ_RESULTS_TABLE(FILE) reads FILE with a comma delimiter and forces
%   the identifier columns to string.
%
%   Both behaviours are needed because readtable's inference gets this wrong
%   in two ways that only show up occasionally:
%
%     * delimiter detection can pick the wrong separator, because level_ranks
%       and rank_by_level are packed with '|' and ':'. On R2024a an 18-column
%       block_size_raw.csv was read as three columns.
%
%     * a git short hash that happens to be all digits, roughly one commit in
%       thirty, is inferred as numeric. Comparing it against a string then
%       errors, and a hash with a leading zero silently loses it, which would
%       let a stale checkpoint or a mismatched shard pass the provenance
%       check that exists precisely to catch them.
    text_columns = {'code_commit', 'case_name', 'rank_pruning', 'method', ...
                    'level_ranks', 'rank_by_level', 'rank_before_pruning_by_level', ...
                    'hostname', 'kernel_type', 'status', 'solver_name', ...
                    'sample_distribution', 'problem', 'config_id'};
    opts = detectImportOptions(file, 'Delimiter', ',', 'TextType', 'string');
    present = intersect(text_columns, opts.VariableNames);
    if ~isempty(present)
        opts = setvartype(opts, present, 'string');
    end
    T = readtable(file, opts);
end
