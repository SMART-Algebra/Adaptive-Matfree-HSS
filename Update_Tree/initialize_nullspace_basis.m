function [Q, state] = initialize_nullspace_basis(method, Omega0, DeltaOmega, options)
%INITIALIZE_NULLSPACE_BASIS Construct the first nullspace block.
%   Omega0 must be square. DeltaOmega contains the first sample batch.

    if nargin < 4 || isempty(options)
        options = struct();
    end

    method = lower(char(method));
    b = size(DeltaOmega, 2);
    if size(Omega0, 1) ~= size(Omega0, 2)
        error('The initial nullspace block Omega0 must be square.');
    end

    state = struct('method', method, 'L', [], 'U', [], 'P', [], ...
        'initial_rcond', NaN, 'target_tolerance', NaN, ...
        'correction_passes', 0, 'residual_threshold', NaN, ...
        'last_nullspace_residual', NaN, 'fallback_active', false, ...
        'recomputed_full', false, 'full_recompute_count', 0, ...
        'hstate', []);

    switch method
        case {'naive', 'global'}
            [Q_full, ~] = qr([Omega0, DeltaOmega]');
            Q = Q_full(:, end-b+1:end);

        case 'householder'
            % Blocked Householder row update of the QR factorization of
            % Omega'. No inverse of Omega0 is formed, so the accuracy is
            % independent of cond(Omega0).
            [Q, state.hstate] = extend_nullspace(Omega0, DeltaOmega);
            state.initial_rcond = rcond(triu(state.hstate.R));

        case 'iterative'
            [L, U, p] = lu(Omega0, 'vector');
            P = -U \ (L \ DeltaOmega(p, :));
            [Q, ~] = qr([P; eye(b, 'like', Omega0)], 'econ');
            state.L = L;
            state.U = U;
            state.P = p;
            state.initial_rcond = rcond(U);

        case 'safeguarded'
            Omega = [Omega0, DeltaOmega];
            [L, U, p] = lu(Omega0, 'vector');
            state.L = L;
            state.U = U;
            state.P = p;
            state.initial_rcond = rcond(U);
            state = configure_correction(state, options, Omega0);

            condition_error = eps(class(Omega0)) / ...
                max(state.initial_rcond, realmin(class(Omega0)));
            if condition_error > state.residual_threshold
                Q = fresh_nullspace_basis(Omega);
                state.last_nullspace_residual = relative_nullspace_residual(Omega, Q);
                state.fallback_active = true;
                state.recomputed_full = true;
                state.full_recompute_count = 1;
            else
                P = -U \ (L \ DeltaOmega(p, :));
                [Q, state.last_nullspace_residual] = ...
                    refine_nullspace_candidate(Omega, ...
                    [P; eye(b, 'like', Omega0)], ...
                    zeros(size(Omega, 2), 0, 'like', Omega0), state);

                if ~isfinite(state.last_nullspace_residual) || ...
                        state.last_nullspace_residual > state.residual_threshold
                    Q = fresh_nullspace_basis(Omega);
                    state.last_nullspace_residual = ...
                        relative_nullspace_residual(Omega, Q);
                    state.fallback_active = true;
                    state.recomputed_full = true;
                    state.full_recompute_count = 1;
                end
            end

        case 'corrected'
            % Correct the fixed-anchor update and trust the measured
            % residual, rather than rejecting a candidate solely from a
            % pessimistic condition-number estimate.
            Omega = [Omega0, DeltaOmega];
            [L, U, p] = lu(Omega0, 'vector');
            state.L = L;
            state.U = U;
            state.P = p;
            state.initial_rcond = rcond(U);
            state = configure_correction(state, options, Omega0);

            P = -U \ (L \ DeltaOmega(p, :));
            [Q, state.last_nullspace_residual] = ...
                refine_nullspace_candidate(Omega, ...
                [P; eye(b, 'like', Omega0)], ...
                zeros(size(Omega, 2), 0, 'like', Omega0), state);

            if ~isfinite(state.last_nullspace_residual) || ...
                    state.last_nullspace_residual > state.residual_threshold
                Q = fresh_nullspace_basis(Omega);
                state.last_nullspace_residual = ...
                    relative_nullspace_residual(Omega, Q);
                state.recomputed_full = true;
                state.full_recompute_count = 1;
            end

        otherwise
            error(['Unknown nullspace method "%s". Choose ''naive'', ' ...
                '''global'', ''iterative'', ''householder'', ' ...
                '''safeguarded'', or ''corrected''.'], method);
    end
end

function state = configure_correction(state, options, Omega0)
    target_tolerance = get_option(options, 'target_tolerance', sqrt(eps(class(Omega0))));
    correction_passes = get_option(options, 'correction_passes', 2);
    residual_fraction = get_option(options, 'residual_fraction', 0.1);

    validateattributes(target_tolerance, {'numeric'}, ...
        {'scalar', 'positive', 'finite'});
    validateattributes(correction_passes, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative', 'finite'});
    validateattributes(residual_fraction, {'numeric'}, ...
        {'scalar', 'positive', 'finite'});

    state.target_tolerance = target_tolerance;
    state.correction_passes = correction_passes;
    state.residual_threshold = max(100 * eps(class(Omega0)), ...
        residual_fraction * target_tolerance);
end

function value = get_option(options, name, default_value)
    if isfield(options, name)
        value = options.(name);
    else
        value = default_value;
    end
end

function Q = fresh_nullspace_basis(Omega)
    nullity = size(Omega, 2) - size(Omega, 1);
    [Q_full, ~] = qr(Omega');
    Q = Q_full(:, end-nullity+1:end);
end

function value = relative_nullspace_residual(Omega, Q)
    value = norm(Omega * Q, 'fro') / max(norm(Omega, 'fro'), realmin(class(Omega)));
end
