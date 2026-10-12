function ok = build_mex(force)
%BUILD_MEX Compile the MEX kernels used by this repository.
%   OK = BUILD_MEX() compiles, into Update_Tree, each of
%     tpqrt_update_mex  (LAPACK DTPQRT/DTPMQRT, extend_nullspace)
%     geqrt_mex         (LAPACK DGEQRT, compute_householder, null_space_basis)
%     gemqrt_mex        (LAPACK DGEMQRT, null_space_basis)
%   that is missing or older than its source, and returns whether all of
%   them are usable. OK = BUILD_MEX(true) rebuilds unconditionally.
%
%   Call this before any timed run. Each kernel has a MATLAB fallback that
%   is correct but far slower, so a run on a machine where the MEX files
%   were never built would report timings that badly understate the method.

    if nargin < 1 || isempty(force), force = false; end

    here = fileparts(mfilename('fullpath'));
    target_dir = fullfile(here, 'Update_Tree');
    names = {'tpqrt_update_mex', 'geqrt_mex', 'gemqrt_mex'};
    ok = true;

    for q = 1:numel(names)
        src = fullfile(target_dir, [names{q} '.c']);
        if exist(src, 'file') ~= 2
            error('build_mex:missingSource', 'Cannot find %s', src);
        end

        target = fullfile(target_dir, [names{q} '.' mexext]);
        % Check the file with isfile, not exist(name, 'file') == 3: exist
        % only finds MEX files whose folder is on the MATLAB path, so it can
        % miss a binary that was just built.
        if ~force && isfile(target)
            d_src = dir(src); d_tgt = dir(target);
            if d_tgt.datenum >= d_src.datenum
                fprintf('build_mex: %s is up to date.\n', target);
                continue;
            end
        end

        fprintf('build_mex: compiling %s ...\n', src);
        try
            mex('-R2018a', src, '-lmwlapack', '-outdir', target_dir);
        catch err
            warning('build_mex:failed', ...
                ['Could not build %s (%s). Its caller will fall back to the ' ...
                 'slower MATLAB code, so do not use timings from this run.'], ...
                names{q}, err.message);
            ok = false;
            continue;
        end
        if isfile(target)
            fprintf('build_mex: built %s\n', target);
        else
            ok = false;
        end
    end

    ok = ok && mex_is_callable(target_dir, names);
end

function ok = mex_is_callable(target_dir, names)
%MEX_IS_CALLABLE Put the MEX directory on the path and check MATLAB sees them.
    addpath(target_dir);
    rehash;
    ok = all(cellfun(@(nm) exist(nm, 'file') == 3, names));
end
