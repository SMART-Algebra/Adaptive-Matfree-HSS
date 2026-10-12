function [Y, b, Omega, original_x, mv, mvT, initial_sketch_time] = build_test_system(n_or_n1d, num_cols, kind, dim, kernel_params, sample_distribution)
    % buildTestSystem:  Generate test data and matvecs for 1D/2D kernels.
    if nargin < 5 || isempty(kernel_params)
        kernel_params = struct();
    end
    if nargin < 6 || isempty(sample_distribution) || strcmpi(sample_distribution, 'auto')
        if strcmpi(kind, 'helmholtz')
            sample_distribution = 'complex';
        else
            sample_distribution = 'real';
        end
    end

    if dim == 1
    % ========================= 1D =========================
        n = n_or_n1d;
        h          = 1/n;
        xlin       = (h : h : 1).';             %  n × 1
        original_x = sin(pi * xlin);            %  n × 1
    
        kind = lower(kind);
    
        if strcmp(kind, 'poisson_schur')
            % Explicit Poisson‐Schur mat‐vec. kernel_params.strip_width
            % (odd; default 51) is the transverse width, a parameter since
            % 2026-10-09; the half-domains are (nx-1)/2 by n.
            nx = get_param(kernel_params, 'strip_width', 51);
            if mod(nx, 2) ~= 1 || nx < 3
                error('build_test_system:strip_width', 'strip_width must be an odd integer >= 3, got %g.', nx);
            end
            mx = (nx - 1) / 2;
            [A, I1, I2, I3] = poisson_2D(nx, n);
            % The half-domains are Dirichlet Laplacians on uniform 25-by-n
            % strips (nx = 51 splits as 25 | separator | 25), so a sine
            % transform along x and tridiagonal solves along y invert them
            % exactly; see DST_STRIP_SOLVER. Test-harness only; the solver
            % itself still sees S through matrix-vector products. Set
            % kernel_params.fast_operator = false for the sparse Cholesky
            % path, e.g. to reproduce earlier timings.
            fast = get_param(kernel_params, 'fast_operator', true);
            if ischar(fast) || isstring(fast)      % 'strip': strip solver on the full blocks
                mv = form_S(A, I1, I2, I3, [mx, n], char(fast));
            elseif fast
                mv = form_S(A, I1, I2, I3, [mx, n]);
            else
                mv = form_S(A, I1, I2, I3);
            end
            mvT = mv;

        elseif strcmp(kind, 'helmholtz_schur')
            % Schur complement of the 2D Helmholtz equation on a 51-by-ny
            % strip, eliminating the two 25-by-ny half-domains and keeping the
            % ny-point separator, exactly as in the Poisson case above.
            %
            % The wavenumber is set by resolution, not by n: we hold
            % kernel_params.points_per_wavelength grid points per wavelength,
            % so k*h is fixed and the domain spans ny/ppw wavelengths. This is
            % the convention used for the 3D Helmholtz box and for the
            % Lippmann-Schwinger problem, and it is what makes the ranks grow
            % with n. An earlier version scaled k as 2*pi*ny/2^15, which holds
            % about 32768 points per wavelength: the shift is then 4e-8
            % against a diagonal of 4, so the operator was numerically the
            % Laplacian and the experiment silently reproduced Poisson.
            ny = n_or_n1d;
            % kernel_params.strip_width (odd; default 51, the manuscript's
            % setting) is the transverse width n_x; the two half-domains are
            % (n_x-1)/2 by n_y. Made a parameter on 2026-10-09 to study
            % wider strips now that the fast operator makes them affordable.
            nx = get_param(kernel_params, 'strip_width', 51);
            if mod(nx, 2) ~= 1 || nx < 3
                error('build_test_system:strip_width', 'strip_width must be an odd integer >= 3, got %g.', nx);
            end
            mx = (nx - 1) / 2;
            ppw = get_param(kernel_params, 'points_per_wavelength', 32);
            h = 1 / (ny + 1);
            k = 2 * pi / (ppw * h);
            shift = (h * k)^2;

            fprintf(['2D Helmholtz FD: nx=%d, ny=%d, %.0f points/wavelength, ' ...
                     'kappa=%.1f, (kh)^2=%.4e\n'], nx, ny, ppw, k / (2 * pi), shift);

            A = fd_helmholtz_strip(nx, ny, k);

            % The grid is nx x ny; we split along the x-axis.
            I1 = 1:((nx-1)/2 * ny);
            I2 = I1 + ((nx+1)/2 * ny);
            I3 = ((nx-1)/2 * ny) + (1:ny);

            % The half-domains are shifted Dirichlet Laplacians on uniform
            % 25-by-ny strips, so the same sine transform used for Poisson
            % inverts them; the shift only moves the mode eigenvalues.
            fast = get_param(kernel_params, 'fast_operator', true);
            if fast
                mv = form_S(A, I1, I2, I3, [mx, ny], [], shift);
            else
                mv = form_S(A, I1, I2, I3, [], [], shift);
            end
            mvT = mv;

        elseif strcmp(kind, 'matern')
            % Matern Kernel: K(r) = (2^(1-nu)/gamma(nu)) * (sqrt(2*nu)*r/ell)^nu * K_nu(...)
            nu  = get_param(kernel_params, 'nu', 2.5);
            ell = get_param(kernel_params, 'ell', 0.1);
            
            indices = (0:n-1)';
            r = indices * h;
            
            t = ones(size(r));
            
            % Compute for r > 0 to avoid singularity in besselk at 0
            mask = r > 0;

            sqrt_2nu = sqrt(2 * nu);
            scaled_r = (sqrt_2nu * r(mask)) / ell;
            
            factor = (2^(1-nu)) / gamma(nu);
            t(mask) = factor .* (scaled_r .^ nu) .* besselk(nu, scaled_r);

            nugget = 1e-2;
            t(1,1) = t(1,1) + nugget^2;

            mv = @(v) toeplitz_matVec_1D(v, t);
            mvT = mv;

        else
            error(['Unknown kind "%s" — must be poisson-schur, ' ...
                                        'helmholtz-schur, or matern.'], kind);
        end

    elseif dim == 2
        % ========================= 2D =========================
        n1d = n_or_n1d;
        h = 1/n1d;
        n = n1d^2;
    
        xx = linspace(h, 1, n1d);
        [X, Ygrid] = meshgrid(xx, xx);
        original_x = sin(pi * X(:)) .* sin(pi * Ygrid(:));
    
        kind = lower(kind);
    
        if strcmp(kind, 'laplace')
            m = 2*n1d - 1;
            C = zeros(m, m);

            K = @(x,y) -(1/(2*pi)) * log(sqrt(x.^2 + y.^2));
            Iquad = integral2(@(x,y) K(x,y), -h/2, h/2, -h/2, h/2);

            for i = 1:m
                for j = 1:m
                    dx = (i - n1d) * h;
                    dy = (j - n1d) * h;
                    if dx == 0 && dy == 0
                        C(i, j) = Iquad;
                    else
                        r = sqrt(dx^2 + dy^2);
                        C(i, j) = h^2 * (-1/(2*pi)) * log(r);
                    end
                end
            end
        
            C = circshift(C, -[n1d-1, n1d-1]);
            
            G = fft2(C);
        
            mv = @(v) toeplitz_matVec_2D(v, G, n1d, 'symmetric');
            mvT = mv;
    
        elseif strcmp(kind, 'poisson_schur')
            % Explicit Poisson‐Schur mat‐vec. kernel_params.strip_width
            % (odd; default 51) is the slab thickness, a parameter since
            % 2026-10-09; the half-boxes are (nx-1)/2 by n1d by n1d.
            nx = get_param(kernel_params, 'strip_width', 51);
            if mod(nx, 2) ~= 1 || nx < 3
                error('build_test_system:strip_width', 'strip_width must be an odd integer >= 3, got %g.', nx);
            end
            mx = (nx - 1) / 2;
            [A, I1, I2, I3] = poisson_3D(nx, n1d, n1d);
            % The half-domains are Dirichlet Laplacians on uniform 25-by-n1d-by-n1d
            % boxes (nx = 51 splits as 25 | separator | 25), so the sine transform
            % diagonalizes them exactly. Test-harness only; the solver itself still
            % sees A through matrix-vector products. Set
            % kernel_params.fast_operator = false to force the sparse
            % factorization instead, e.g. to reproduce earlier timings.
            if get_param(kernel_params, 'fast_operator', true)
                mv = form_S_2D(A, I1, I2, I3, [mx, n1d, n1d]);
            else
                mv = form_S_2D(A, I1, I2, I3);
            end
            mvT = mv;
        
        elseif strcmp(kind, 'helmholtz_schur')    
            nx = 51; 
            ny = n1d; 
            nz = n1d;
            
            kappa = n1d / 32; 
            k = 2 * pi * kappa;

            fprintf('3D Helmholtz FD: %dx%dx%d, k=%.2f\n', nx, ny, nz, k);
            
            % Construct 3D Laplacian and index sets
            [L, I1, I2, I3] = poisson_3D(nx, ny, nz);
            
            % Apply Helmholtz shift: (-Delta - k^2) u = f
            % Discretized: (1/h^2 * L - k^2 I) u = f
            % Multiplied by h^2: (L - (kh)^2 I) u = h^2 f
            h = 1 / (n1d + 1);
            shift = (k * h)^2;
            
            N_tot = size(L, 1);
            A = L - shift * speye(N_tot);
            
            % Form Schur complement
            mv = form_S_2D(A, I1, I2, I3);
            mvT = mv;
    
        elseif strcmp(kind, 'helmholtz')
            kappa = n1d/32;
            k = 2*pi*kappa;
            n = n1d^2;
            h = 1 / n1d;
        
            % Flatten
            x = [X(:)'; Ygrid(:)'];

            % POTENTIAL is 'cavity' (the paper's Section 5.5 experiment:
            % the near-resonant cavity scatterer of Gopal & Martinsson
            % 2022, arXiv:2007.12718, Section 6.2) or 'gaussian_bump'. The
            % default stays 'gaussian_bump' for backward compatibility; the
            % paper's drivers pass 'cavity' explicitly.
            potential = get_param(kernel_params, 'potential', 'gaussian_bump');
            xc = x(1,:) - 0.5;
            yc = x(2,:) - 0.5;

            switch lower(potential)
                case 'gaussian_bump'
                    % --- Gaussian Bump Potential ---
                    r_sq = xc.^2 + yc.^2;
                    V = exp(-32 * r_sq); % b(x) = exp(-32*||x-c||^2)
                case 'cavity'
                    % --- Cavity Potential (arXiv:2007.12718, eq. in Sec. 6.2) ---
                    %   b(r,theta) = (1 - sin(0.5 theta)^500) exp(-2000 (0.1 - r^2)^2)
                    % A near-resonant cavity: the sin^500 term opens a thin
                    % slot in an otherwise near-total reflector ring at
                    % r^2 = 0.1, which the paper reports makes
                    % unpreconditioned GMRES "unfeasibly high" even at
                    % modest frequency.
                    r = sqrt(xc.^2 + yc.^2);
                    theta = atan2(yc, xc);
                    sin_term = (sin(0.5 * theta)).^500;
                    exp_term = exp(-2000 * (0.1 - r.^2).^2);
                    V = (1 - sin_term) .* exp_term;
                otherwise
                    error('build_test_system:potential', ...
                        'Unknown potential "%s"; use ''gaussian_bump'' or ''cavity''.', ...
                        potential);
            end

            sqrtb = k * sqrt(V(:));
        
            intgrnd = @(x,y)(0.25i * besselh(0,1,k * sqrt(x.^2 + y.^2)));
            intgrl = integral2(intgrnd, -h/2, h/2, -h/2, h/2);
        
            grid_vec = (0:n1d-1)*h;
            [x_dist, y_dist] = ndgrid(grid_vec, grid_vec);
            r = sqrt(x_dist.^2 + y_dist.^2);
            t = (1i/4) * besselh(0,1,k*r) * h^2;
            t(1,1) = intgrl; % handle the diagonal
        
            % Embed into 2D circulant block
            T = reshape(t, n1d, n1d);
            B = zeros(2*n1d-1, 'like', T);
            B(1:n1d,1:n1d) = T;
            B(1:n1d,n1d+1:end) = T(:,2:end);
            B(n1d+1:end,1:n1d) = T(2:end,:);
            B(n1d+1:end,n1d+1:end) = T(2:end,2:end);
            B(n1d+1:end,:) = flip(B(n1d+1:end,:),1);
            B(:,n1d+1:end) = flip(B(:,n1d+1:end),2);
        
            % FFT
            G = fft2(B);
        
            % This operator is complex symmetric, not Hermitian: A.' = A
            % but A' = conj(A). The factorization needs A.' (every basis
            % change is a congruence U.'*A*U), so it takes the symmetric
            % path and never calls mvT -- 'helmholtz' is in
            % symmetric_kernel_list. mvT below is the conjugate transpose,
            % which is what cond2_from_mv wants for A'*A; it is NOT a
            % substitute for A.' in a basis pass. Removing this kernel from
            % symmetric_kernel_list would therefore build V from the wrong
            % operator.
            mv  = @(v) toeplitz_matVec_2D(v, G, n1d, 'helmholtz', sqrtb);
            mvT = @(v) toeplitz_matVec_2D(v, conj(G), n1d, 'helmholtz', sqrtb);
        
        elseif strcmp(kind, 'matern')
            % Paper (Sec. 7.2) reports two parameter sets; set nu, ell to
            % reproduce the desired table:
            %   Table 5: nu = 0.5, ell = 0.5
            %   Table 6: nu = 1.5, ell = 0.1
            nu  = get_param(kernel_params, 'nu', 0.5);
            ell = get_param(kernel_params, 'ell', 0.5);

            % Distance grid for the first block-row
            grid_vec = (0:n1d-1)*h;
            [x_dist, y_dist] = ndgrid(grid_vec, grid_vec);
            r = sqrt(x_dist.^2 + y_dist.^2);

            % Initialize t with 1.0  (at r = 0)
            t = ones(size(r));

            mask = r > 0;
            
            if any(mask(:))
                sqrt_2nu = sqrt(2 * nu);
                scaled_r = (sqrt_2nu * r(mask)) / ell;

                factor = (2^(1-nu)) / gamma(nu);
                t(mask) = factor .* (scaled_r .^ nu) .* besselk(nu, scaled_r);
            end

            nugget = 1e-2;   % adds nugget^2 = 1e-4 to the diagonal (sigma^2 in the paper)
            t(1,1) = t(1,1) + nugget^2;

            % Embed into 2D circulant block
            T = t;
            B = zeros(2*n1d-1, 'like', T);
            B(1:n1d, 1:n1d) = T;
            B(1:n1d, n1d+1:end) = T(:, 2:end);
            B(n1d+1:end, 1:n1d) = T(2:end, :);
            B(n1d+1:end, n1d+1:end) = T(2:end, 2:end);
            B(n1d+1:end, :) = flip(B(n1d+1:end, :), 1);
            B(:, n1d+1:end) = flip(B(:, n1d+1:end), 2);
            G = fft2(B);

            mv = @(v) toeplitz_matVec_2D(v, G, n1d, 'symmetric');
            mvT = mv;

        else
            error(['Unknown kind "%s" — must be laplace, poisson-schur, ' ...
                                        'helmholtz-schur, or matern.'], kind);
        end
    end

    sketch_timer = tic;
    [Omega, Y] = add_columns_at_leaves(num_cols, mv, n, sample_distribution);
    initial_sketch_time = toc(sketch_timer);
    b = mv(original_x);
end

function value = get_param(params, name, default_value)
    if isfield(params, name)
        value = params.(name);
    else
        value = default_value;
    end
end
