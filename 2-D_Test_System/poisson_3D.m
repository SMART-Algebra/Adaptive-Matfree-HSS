function [A, I1, I2, I3] = poisson_3D(nx, ny, nz)
% Builds a 3-D Poisson matrix A for an nx x ny x nz grid using finite differences.
%
% Input:
%   nx - number of grid points in the x-direction. (Odd number. I chose 51.)
%   ny - number of grid points in the y-direction.
%   nz - number of grid points in the z-direction.
%
% Output:
%   A  - The 3-D Poisson matrix of size (nx*ny*nz) x (nx*ny*nz).
%        This corresponds to a 7-point stencil discretization of -Δu = f.
%
%   I1, I2, I3 - Index sets for partitioning the rows of A

    fprintf("At poisson3D...\n");

    Lx = spdiags(repmat([-1 2 -1], nx, 1), -1:1, nx, nx); 
    Ly = spdiags(repmat([-1 2 -1], ny, 1), -1:1, ny, ny); 
    Lz = spdiags(repmat([-1 2 -1], nz, 1), -1:1, nz, nz); 

    Ix = speye(nx);
    Iy = speye(ny);
    Iz = speye(nz);

    % Construct 3D Poisson:
    % A = I_x ⊗ I_y ⊗ Lz + I_x ⊗ Ly ⊗ I_z + Lx ⊗ I_y ⊗ I_z
    A = kron(Ix, kron(Iy, Lz)) + ...
        kron(Ix, kron(Ly, Iz)) + ...
        kron(Lx, kron(Iy, Iz));

    % Determine slice size (number of unknowns per x-slice)
    ny_nz = ny * nz;

    I1 = 1:((nx-1)/2 * ny_nz);      
    I2 = I1 + ((nx+1)/2 * ny_nz);    
    I3 = ((nx-1)/2 * ny_nz) + (1:ny_nz);
end
