function [DeltaQ, state] = extend_nullspace(state, DeltaOmega)
%EXTEND_NULLSPACE Append nullspace directions by a Householder QR row update.
%   [DeltaQ, STATE] = EXTEND_NULLSPACE(OMEGA0, DELTAOMEGA) initializes the
%   update from the square test block OMEGA0 (m-by-m) and returns the first
%   b = size(DELTAOMEGA,2) nullspace directions of [OMEGA0, DELTAOMEGA].
%
%   [DeltaQ, STATE] = EXTEND_NULLSPACE(STATE, DELTAOMEGA) appends a further
%   batch. DeltaQ has size (p+b)-by-b, where p is the number of columns of
%   the test matrix before the batch.
%
%   The retained basis is preserved exactly. If Q_old spans ker(Omega_old),
%   then [Q_old; zeros(b,k)] spans the corresponding directions of
%   ker(Omega_new), so
%
%       Q_new = [Q_old; zeros(b,k)],  DeltaQ]
%
%   and previously computed samples Y_old*Q_old remain valid unchanged.
%
%   ker(Omega) is the trailing block of the orthogonal factor of Omega'.
%   Appending b columns to Omega appends b rows to Omega', which is the
%   classical QR row update: reflector j acts only on pivot row j and the b
%   appended rows, so it never touches the coordinates carrying the retained
%   nullspace. Every operation is orthogonal, so no inverse of Omega0 enters
%   and the accuracy does not depend on cond(Omega0).
%
%   Uses the LAPACK kernels DTPQRT/DTPMQRT (ZTPQRT/ZTPMQRT for complex input)
%   through TPQRT_UPDATE_MEX when that MEX file is built, and an equivalent
%   unblocked Householder loop otherwise. Build the MEX with
%
%       mex -R2018a tpqrt_update_mex.c -lmwlapack
%
%   Complex input is supported. For complex Omega the relevant factorization
%   is that of Omega^H, so conjugate transposes are used throughout.
%
%   See also INITIALIZE_NULLSPACE_BASIS, EXTEND_NULLSPACE_BASIS.

    if isnumeric(state)
        Omega0 = state;
        if size(Omega0, 1) ~= size(Omega0, 2)
            error('extend_nullspace:shape', 'Omega0 must be square.');
        end
        [Qf, R] = qr(Omega0');
        state = struct('R', triu(R), 'Q1', Qf, 'use_mex', have_mex());
    end

    validateattributes(DeltaOmega, {'double'}, {'2d', 'nonempty'});
    m = size(state.R, 1);
    if size(DeltaOmega, 1) ~= m
        error('extend_nullspace:shape', ...
            'DeltaOmega must have %d rows.', m);
    end

    % The MEX kernel requires the three arrays to share a type.
    if ~isreal(state.Q1) || ~isreal(state.R) || ~isreal(DeltaOmega)
        state.R = complex(state.R);
        state.Q1 = complex(state.Q1);
        DeltaOmega = complex(DeltaOmega);
    end

    if state.use_mex
        [state.R, state.Q1, DeltaQ] = ...
            tpqrt_update_mex(state.R, DeltaOmega', state.Q1);
    else
        [state.R, state.Q1, DeltaQ] = ...
            householder_row_update(state.R, DeltaOmega', state.Q1);
    end
end

% ------------------------------------------------------------------------
function tf = have_mex()
    persistent cached
    if isempty(cached)
        cached = exist('tpqrt_update_mex', 'file') == 3;
        if ~cached
            % The fallback is correct but much slower. Say so once, so a
            % timed run cannot silently report the wrong implementation.
            warning('extend_nullspace:noMex', ...
                ['tpqrt_update_mex is not built; using the pure-MATLAB ' ...
                 'Householder loop, which is substantially slower. Run ' ...
                 'build_mex before any timed run.']);
        end
    end
    tf = cached;
end

% ------------------------------------------------------------------------
function [R, Q1new, DeltaQ] = householder_row_update(R, B, Q1)
%HOUSEHOLDER_ROW_UPDATE Unblocked reference for one row update of [R; B].
%   Identical arithmetic to DTPQRT/DTPMQRT with block size 1.

    [b, m] = size(B);
    p = size(Q1, 1);
    cols = [[Q1; zeros(b, m)], [zeros(p, b); eye(b)]];
    newidx = m + 1 : m + b;

    for j = 1:m
        x = [R(j, j); B(:, j)];
        nrm = norm(x);
        if nrm == 0
            continue;
        end
        v = x;
        v(1) = v(1) + sign_nz(x(1)) * nrm;
        nv = norm(v);
        if nv == 0
            continue;
        end
        v = v / nv;

        blk = [R(j, j:m); B(:, j:m)];          % apply on the left
        blk = blk - (2 * v) * (v' * blk);
        R(j, j:m) = blk(1, :);
        B(:, j:m) = blk(2:end, :);

        idx = [j, newidx];                     % apply on the right
        sub = cols(:, idx);
        cols(:, idx) = sub - (sub * v) * (2 * v');
    end

    R = triu(R);
    Q1new = cols(:, 1:m);
    DeltaQ = cols(:, m+1:end);
end

% ------------------------------------------------------------------------
function s = sign_nz(x)
%SIGN_NZ Unit-modulus phase of x, returning 1 at x = 0.
%   Reduces to +/-1 for real input and gives the standard complex
%   Householder choice otherwise.
    if x == 0
        s = 1;
    else
        s = x / abs(x);
    end
end
