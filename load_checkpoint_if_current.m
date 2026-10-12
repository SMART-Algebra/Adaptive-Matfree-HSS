function [saved, resumed] = load_checkpoint_if_current(checkpoint_file, companions)
%LOAD_CHECKPOINT_IF_CURRENT Load a checkpoint only if the current code wrote it.
%   [SAVED, RESUMED] = LOAD_CHECKPOINT_IF_CURRENT(FILE, COMPANIONS) returns
%   the checkpoint's variables in SAVED and RESUMED = true when FILE exists
%   and its code_commit field equals code_commit_hash(). A checkpoint
%   without that field, or with another commit, is moved aside as
%   <name>.stale-<commit><ext>, and so is every existing file in the cell
%   array COMPANIONS (the CSVs and .mat files the same run wrote); SAVED is
%   then [] and RESUMED false, so the caller starts from scratch.
%
%   Setting the environment variable HSS_RESUME_ACROSS_COMMITS=1 resumes a
%   mismatched checkpoint anyway, for the case where the code changed in a
%   way that does not affect the results; the mismatch is still printed.
%
%   Why: on 2026-09-12 a resubmitted studies task "resumed 45 completed
%   runs" from a checkpoint an earlier submission had left in the results
%   directory, skipped every configuration, and left CSVs computed by code
%   with two bugs that had been fixed in between.
    if nargin < 2 || isempty(companions), companions = {}; end
    saved = [];
    resumed = false;
    if ~isfile(checkpoint_file), return; end
    saved = load(checkpoint_file);
    current = code_commit_hash();
    if isfield(saved, 'code_commit')
        written_by = char(string(saved.code_commit));
    else
        written_by = 'unstamped';
    end
    if strcmp(written_by, current)
        resumed = true;
        return;
    end
    fprintf(['\nCheckpoint %s was written by code %s; the checked-out code ' ...
        'is %s.\n'], checkpoint_file, written_by, current);
    if strcmp(getenv('HSS_RESUME_ACROSS_COMMITS'), '1')
        fprintf('HSS_RESUME_ACROSS_COMMITS=1 is set, so it is resumed anyway.\n');
        resumed = true;
        return;
    end
    set_aside(checkpoint_file, written_by);
    for q = 1:numel(companions)
        if isfile(companions{q}), set_aside(companions{q}, written_by); end
    end
    fprintf(['It and the files it produced were renamed *.stale-%s.* and this ' ...
        'run starts from scratch. Set HSS_RESUME_ACROSS_COMMITS=1 to resume ' ...
        'across commits knowingly.\n'], written_by);
    saved = [];
    resumed = false;
end

function set_aside(file, tag)
    [d, base, ext] = fileparts(file);
    aside = fullfile(d, sprintf('%s.stale-%s%s', base, tag, ext));
    k = 1;
    while isfile(aside)
        k = k + 1;
        aside = fullfile(d, sprintf('%s.stale-%s-%d%s', base, tag, k, ext));
    end
    movefile(file, aside);
    fprintf('  moved %s -> %s\n', file, aside);
end
