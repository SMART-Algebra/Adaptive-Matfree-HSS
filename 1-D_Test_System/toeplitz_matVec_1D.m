function y = toeplitz_matVec_1D(x, t)
    % toeplitzMatVec:
    %    Multiplies the symmetric Toeplitz matrix (defined by 't')
    %    by the vector x via FFT-based convolution in O(n log n) time.
    %
    % Inputs:
    %    x : n x 1 vector
    %    t : n x 1 vector - the first row of the Toeplitz matrix
    %
    % Output:
    %    y : n x 1, the product T * x

    n = length(x);

    % --- Build the circulant embedding 'c' of length 2n ---
    %   The first n entries are t,
    %   Then append the reverse of t(2..n).
    c = zeros(2*n, 1);
    c(1:n) = t;
    tail = t(n:-1:2);        % reversed portion of length (n-1)
    c(n+2 : 2*n) = tail;     % due to symmetric A_i

    % Zero-pad x to length 2n
    x_pad = [x; zeros(n,1)];

    % --- Convolution via FFT ---
    fft_c = fft(c);
    fft_x = fft(x_pad);
    conv_result = ifft( fft_c .* fft_x );

    % --- The first n entries is the Toeplitz product ---
    y = conv_result(1:n);
end
