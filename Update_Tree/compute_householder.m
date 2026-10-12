function [W, C] = compute_householder(A)
    % Computes the WY representation of Householder reflectors.
    % Q = I - W * C'
    %
    % Input: 
    %   A: (m x n) matrix to factorize
    % Output: 
    %   W: (m x n) The combined weight matrix
    %   C: (m x n) The matrix of Householder vectors
    %
    % Fast path: if m >= n >= 1 and GEQRT_MEX is built (build_mex), use
    % LAPACK DGEQRT for real A and ZGEQRT for complex A. Same result as the
    % loop below up to rounding and a unit-modulus factor per column (a
    % sign when real). Otherwise, use the loop below.

    [m, n] = size(A);
    if m >= n && n >= 1 && have_mex()
        [C, T] = geqrt_mex(A);
        W = C * triu(T);
        return;
    end
    C = zeros(m, n, 'like', A);
    tau = zeros(1, n, 'like', A);
    
    work_A = A;
    
    % --- Phase 1: Compute V and tau ---
    for k = 1:n
        x = work_A(k:m, k);
        norm_x = norm(x);
        
        if norm_x == 0
            v = x; 
            tau(k) = 0;
        else
            s = sign(x(1));
            if s == 0, s = 1; end
            
            v = x;
            v(1) = x(1) + s * norm_x;
            v = v / v(1); 
            
            beta = 2 / (v' * v);
            tau(k) = beta;
            
            if k < n
                w = beta * (work_A(k:m, k+1:n)' * v);
                work_A(k:m, k+1:n) = work_A(k:m, k+1:n) - v * w';
            end
        end
        C(k:m, k) = v;
    end
    
    % --- Phase 2: Compute T and then W ---
    % Q = I - C * T * C'  -> Want Q = I - W * C'
    % So, W = C * T
    
    T = zeros(n, n, 'like', A);
    if n > 0
        T(1, 1) = tau(1);
        for i = 2:n
            beta = tau(i);
            v_prev = C(:, 1:i-1);
            v_curr = C(:, i);
            
            % T recursion: T(1:i-1, i) = -tau * T * (V_prev' * v_curr)
            prev_interactions = v_prev' * v_curr;
            T(1:i-1, i) = -beta * (T(1:i-1, 1:i-1) * prev_interactions);
            T(i, i) = beta;
        end
    end
    
    % Compute W
    W = C * T;
end

% ------------------------------------------------------------------------
function tf = have_mex()
    persistent cached
    if isempty(cached)
        cached = exist('geqrt_mex', 'file') == 3;
        if ~cached
            % The fallback is correct but much slower. Say so once, so a
            % timed run cannot silently report the wrong implementation.
            warning('compute_householder:noMex', ...
                ['geqrt_mex is not built; using the interpreted Householder ' ...
                 'loop, which is substantially slower. Run build_mex before ' ...
                 'any timed run.']);
        end
    end
    tf = cached;
end
