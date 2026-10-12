function hat_Q = null_space_basis(Omega_i, k)
%NULL_SPACE_BASIS Last k columns of the full Q of qr(Omega_i'), Omega_i m x p.
%   For p >= m + k these are k orthonormal directions with
%   Omega_i * hat_Q = 0, the test directions U_function uses.
%
%   For p >= m >= 1 with GEQRT_MEX and GEMQRT_MEX built (build_mex), Q
%   stays in compact WY form (LAPACK DGEQRT, or ZGEQRT when Omega_i is
%   complex) and only Q * [0; I_k] is formed (DGEMQRT/ZGEMQRT): about
%   4*p*m*k flops instead of the 4*(p^2*m - p*m^2 + m^3/3) of forming the
%   whole p x p Q. Same reflectors as MATLAB's qr, so the columns agree up
%   to rounding. Otherwise this forms the full Q with qr.

    [m, p] = size(Omega_i);
    if p >= m && m >= 1 && have_mex()
        nb = min(m, 32);
        [V, T] = geqrt_mex(Omega_i', nb);
        E = [zeros(p - k, k); eye(k)];
        if ~isreal(V), E = complex(E); end
        hat_Q = gemqrt_mex(V, T, E, 'N');
    else
        [Q, ~] = qr(Omega_i');
        hat_Q = Q(:, end - k + 1:end);
    end
end

% ------------------------------------------------------------------------
function tf = have_mex()
    persistent cached
    if isempty(cached)
        cached = exist('geqrt_mex', 'file') == 3 && exist('gemqrt_mex', 'file') == 3;
        if ~cached
            warning('null_space_basis:noMex', ...
                ['geqrt_mex/gemqrt_mex are not built; forming the full Q, ' ...
                 'which is substantially slower. Run build_mex before any ' ...
                 'timed run.']);
        end
    end
    tf = cached;
end
