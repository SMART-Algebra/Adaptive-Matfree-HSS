function A = fd_helmholtz_strip(nx, ny, k)
% FD_HELMHOLTZ_STRIP  Builds Helmholtz matrix on a rectangle (nx x ny).
%
% Inputs:
%   nx - Grid points in x-direction (width)
%   ny - Grid points in y-direction (height/interface length)
%   k  - Wavenumber (default: 2*pi*8)
%
% Output:
%   A  - Sparse matrix of size (nx*ny) x (nx*ny)

    if nargin < 3, k = 2*pi*8; end

    h = 1/(ny+1); 
    
    % Laplacian Stencils ([-1, 2, -1])
    Lx = spdiags(ones(nx,1)*[-1 2 -1], -1:1, nx, nx);
    Ly = spdiags(ones(ny,1)*[-1 2 -1], -1:1, ny, ny);
    
    % Identity Matrices
    Ix = speye(nx);
    Iy = speye(ny);
    
    % Build 2D Laplacian using Kronecker Sum
    L = kron(Lx, Iy) + kron(Ix, Ly);
    
    % 4. Apply Helmholtz Shift (-k^2)
    % The standard FD Laplacian is (4*u - sum(neighbors))/h^2.
    % The matrix L above provides the "4*u - sum(neighbors)" part.
    % The equation is: (L/h^2 - k^2 I) u = f
    % Multiply by h^2: (L - h^2*k^2 I) u = h^2 f
    
    N = nx * ny;
    shift = (h^2 * k^2);
    
    A = L - shift * speye(N);
    
end