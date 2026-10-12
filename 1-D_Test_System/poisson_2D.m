% For Poisson kernel

function [A, I1, I2, I3] = poisson_2D(nx, ny)
% Builds a 2-D Poisson matrix A for a nx x ny grid.
%
% Input:
%   nx - number of grid points in the x-direction. (Odd number. I chose 51)
%   ny - number of grid points in the y-direction. (Determines the size of S)
%
% Output:
%   A  - The 2-D Poisson matrix of size (nx*ny) x (nx*ny).
%   I1, I2, I3 - Index sets for partitioning the rows.

    Lx = spdiags(repmat([-1 2 -1], nx, 1), -1:1, nx, nx);
    Ly = spdiags(repmat([-1 2 -1], ny, 1), -1:1, ny, ny);
    
    A = kron(Lx, speye(ny)) + kron(speye(nx), Ly);
    
    I1 = 1:((nx-1)/2 * ny);        % For nx=3, I1 = 1:ny.
    I2 = I1 + ((nx+1)/2 * ny);     % For nx=3, I2 = (1:ny) + 2*ny.
    I3 = ((nx-1)/2 * ny) + (1:ny); % For nx=3, I3 = ny + (1:ny).
end
