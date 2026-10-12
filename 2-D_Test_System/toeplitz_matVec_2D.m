function y = toeplitz_matVec_2D(x, G, n1d, kind, sqrtb)
% toeplitzMatVec2D:   
%   Compute y = A * x via embedding in a 2D circulant and FFT.
%
% Inputs:
%   x     : n1d^2 × 1 vector
%   G     : 2D FFT of the circulant embedding (size (2n1d-1)^2)
%   noise : diagonal value to add (e.g. 2 for A(ii)=2)
%
% Output:
%   y     : n1d^2 × 1 vector, the product A * x


    switch lower(kind)
        case 'symmetric'
            X = reshape(x, n1d, n1d);
           
            P = zeros(2*n1d-1, 'like', X);
            P(1:n1d, 1:n1d) = X;
            
            % Convolution in Fourier domain
            Conv = ifft2(G .* fft2(P));
            
            % Extract the top‐left block and flatten
            y = Conv(1:n1d, 1:n1d);
            y = y(:);                       

        case 'helmholtz'
            if nargin < 5 || isempty(sqrtb)
                error('sqrtb (n1d×n1d) must be provided for Helmholtz kernel.');
            end

            sqrtb_vec = sqrtb(:); 
            X = reshape(sqrtb_vec .* x, n1d, n1d); % apply sqrtb

            P = zeros(2*n1d-1, 'like', X); % zero-pad to embedding size
            P(1:n1d, 1:n1d) = X;
        
            Conv = ifft2(G .* fft2(P));
        
            y = sqrtb_vec .* reshape(Conv(1:n1d, 1:n1d), n1d^2, 1);
            y = y + x;

        otherwise
            error('Unknown kernel type "%s". Use "symmetric" or "helmholtz".', kind);
    end
end
