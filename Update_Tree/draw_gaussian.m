function Omega = draw_gaussian(n, num_cols, distribution)
%DRAW_GAUSSIAN Draw real or circular complex standard Gaussian samples.
%   distribution is 'real' or 'complex'. A complex sample has entries
%   (g1 + 1i*g2)/sqrt(2), so E|Omega_ij|^2 = 1.

    if nargin < 3 || isempty(distribution)
        distribution = 'real';
    end
    distribution = lower(char(distribution));

    switch distribution
        case 'real'
            Omega = randn(n, num_cols);
        case 'complex'
            Omega = (randn(n, num_cols) + 1i * randn(n, num_cols)) / sqrt(2);
        otherwise
            error('draw_gaussian:InvalidDistribution', ...
                'Unknown sample distribution "%s". Choose ''real'' or ''complex''.', ...
                distribution);
    end
end
