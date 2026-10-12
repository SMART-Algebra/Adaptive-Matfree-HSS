function hash = code_commit_hash()
%CODE_COMMIT_HASH Short git commit of the checked-out code, or 'unknown'.
%   Used to stamp checkpoints and result rows so that a checkpoint written
%   by an older version of the code is never resumed as if it were current:
%   on 2026-09-12 a studies job "resumed 45 completed runs" from a
%   checkpoint left by an earlier submission and skipped every
%   configuration, and the CSVs it left behind carried two bugs that had
%   been fixed in between.
    here = fileparts(mfilename('fullpath'));
    hash = 'unknown';
    try
        [status, out] = system(sprintf('git -C "%s" rev-parse --short HEAD', here));
        out = strtrim(out);
        if status == 0 && ~isempty(out) && all(isstrprop(out, 'xdigit'))
            hash = out;
        end
    catch
    end
end
