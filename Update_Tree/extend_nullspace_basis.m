function [Q, DeltaQ, state] = extend_nullspace_basis( ...
        method, Omega, Q_prev, state, DeltaOmega)
%EXTEND_NULLSPACE_BASIS Add one nullspace block while retaining Q_prev.

    method = lower(char(method));
    b = size(DeltaOmega, 2);
    k_prev = size(Q_prev, 2);
    Q_padded = [Q_prev; zeros(b, k_prev, 'like', Q_prev)];
    state.recomputed_full = false;

    switch method
        case 'naive'
            % Recompute the complete nullspace from all accumulated samples.
            % The caller also rebuilds the complete off-diagonal basis.
            [Q_full, ~] = qr(Omega');
            nullity = size(Omega, 2) - size(Omega, 1);
            Q = Q_full(:, end-nullity+1:end);
            DeltaQ = Q(:, end-b+1:end);

        case 'global'
            [Q_full, ~] = qr(Omega');
            Q_null = Q_full(:, end-(k_prev+b)+1:end);

            % Choose the complement that preserves the retained basis.
            [W, ~] = qr(Q_null' * Q_padded);
            DeltaQ = Q_null * W(:, end-b+1:end);

        case 'iterative'
            if isempty(state.U)
                error('The iterative nullspace state has not been initialized.');
            end
            P = -state.U \ (state.L \ DeltaOmega(state.P, :));
            candidate = [P; zeros(k_prev, b, 'like', P); eye(b, 'like', P)];
            DeltaQ = candidate - Q_padded * (Q_padded' * candidate);

        case 'householder'
            % One blocked Householder row update. The reflectors never touch
            % the coordinates holding the retained basis, so Q_padded is
            % reproduced exactly and only the new block is computed.
            [DeltaQ, state.hstate] = extend_nullspace(state.hstate, DeltaOmega);
            Q = [Q_padded, DeltaQ];
            return;

        case 'safeguarded'
            if isempty(state.U)
                error('The safeguarded nullspace state has not been initialized.');
            end

            if state.fallback_active
                Q = fresh_nullspace_basis(Omega);
                DeltaQ = Q(:, end-b+1:end);
                state.recomputed_full = true;
                state.full_recompute_count = state.full_recompute_count + 1;
                state.last_nullspace_residual = ...
                    relative_nullspace_residual(Omega, Q);
                return;
            end

            P = -state.U \ (state.L \ DeltaOmega(state.P, :));
            candidate = [P; zeros(k_prev, b, 'like', P); eye(b, 'like', P)];
            [DeltaQ, state.last_nullspace_residual] = ...
                refine_nullspace_candidate(Omega, candidate, Q_padded, state);
            Q = [Q_padded, DeltaQ];

            if ~isfinite(state.last_nullspace_residual) || ...
                    state.last_nullspace_residual > state.residual_threshold
                Q = fresh_nullspace_basis(Omega);
                DeltaQ = Q(:, end-b+1:end);
                state.fallback_active = true;
                state.recomputed_full = true;
                state.full_recompute_count = state.full_recompute_count + 1;
                state.last_nullspace_residual = ...
                    relative_nullspace_residual(Omega, Q);
            end
            return;

        case 'corrected'
            if isempty(state.U)
                error('The corrected nullspace state has not been initialized.');
            end

            P = -state.U \ (state.L \ DeltaOmega(state.P, :));
            candidate = [P; zeros(k_prev, b, 'like', P); eye(b, 'like', P)];
            [DeltaQ, state.last_nullspace_residual] = ...
                refine_nullspace_candidate(Omega, candidate, Q_padded, state);
            Q = [Q_padded, DeltaQ];

            if ~isfinite(state.last_nullspace_residual) || ...
                    state.last_nullspace_residual > state.residual_threshold
                Q = fresh_nullspace_basis(Omega);
                DeltaQ = Q(:, end-b+1:end);
                state.recomputed_full = true;
                state.full_recompute_count = state.full_recompute_count + 1;
                state.last_nullspace_residual = ...
                    relative_nullspace_residual(Omega, Q);
            end
            return;

        otherwise
            error(['Unknown nullspace method "%s". Choose ''naive'', ' ...
                '''global'', ''iterative'', ''householder'', ' ...
                '''safeguarded'', or ''corrected''.'], method);
    end

    if strcmp(method, 'naive')
        return;
    end

    % Reorthogonalize the new block against the retained basis.
    [DeltaQ, ~] = qr(DeltaQ, 'econ');
    DeltaQ = DeltaQ - Q_padded * (Q_padded' * DeltaQ);
    [DeltaQ, ~] = qr(DeltaQ, 'econ');
    Q = [Q_padded, DeltaQ];
end

function Q = fresh_nullspace_basis(Omega)
    nullity = size(Omega, 2) - size(Omega, 1);
    [Q_full, ~] = qr(Omega');
    Q = Q_full(:, end-nullity+1:end);
end

function value = relative_nullspace_residual(Omega, Q)
    value = norm(Omega * Q, 'fro') / ...
        max(norm(Omega, 'fro'), realmin(class(Omega)));
end
