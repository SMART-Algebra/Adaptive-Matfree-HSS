function [Y2, b2, Omega2, original_x, G] = test_system_toeplitz_2D(n1d, pt_leaf, k_A)
    % test2DToeplitzMatVec  
    %   Builds (conceptually) the 2-D Toeplitz matrix A where
    %   where
    %       A(r,c) = 1/dist  , normalized by 1/(n1d^2),
    %       A(r,r) = 2,
    %    but avoids forming it explicitly.
    %
    %   Then computes:
    %     Y2 = A * Omega2
    %     b2 = A * original_x
    %
    % Inputs:
    %   n1d         : number of points per side (total points = n1d^2)
    %   pt_leaf     : number of points per leaf (extra columns for Omega_i)
    %   k_A         : chosen rank (extra columns for Omega_i)
    %
    % Outputs:
    %   Y2             : n_i x max(pt_leaf + k_A, 5 * k_A) matrix
    %   b2             : n_i × 1 vector
    %   Omega2         : random n_i x max(pt_leaf + k_A, 5 * k_A) test matrix
    %   original_x     : 2-D sinusoidal test vector
    %   G              : 2-D FFT of the circulant embedding (for reuse)

  rng(0);
  
  h = 1/n1d;
  n = n1d^2;

  %--- Build flattened grid for original_x
  xx = linspace(h,1,n1d);
  [X,Y] = meshgrid(xx,xx);
  Xf = X(:);
  Yf = Y(:);
  original_x = sin(pi*Xf).*sin(pi*Yf);

  %--- Build circulant embedding of the kernel ---
  m = 2*n1d - 1;
  C = zeros(m,m);
  for i = 1:m
    for j = 1:m
      du = (i-1-(n1d-1)) * h;
      dv = (j-1-(n1d-1)) * h;
      if du == 0 && dv == 0
        C(i,j) = 0;
      else
        C(i,j) = 1/sqrt(du^2 + dv^2);
      end
    end
  end

  C = C / n; % normalize A = A/n1d^2

  C(n1d, n1d) = C(n1d, n1d) + 2; % add noise = 2
  C = circshift(C, -[n1d-1, n1d-1]);

  % Precompute 2D FFT of the embedding
  G = fft2(C);

  Omega2 = randn(n, max(pt_leaf + k_A, 5 * k_A));

  Y2 = zeros(n, size(Omega2,2));
  for col = 1:size(Omega2,2)
    Y2(:,col) = toeplitz_matVec_2D( Omega2(:,col), G, n1d);
  end

  b2 = toeplitz_matVec_2D(original_x, G, n1d);

end
