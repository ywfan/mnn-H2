function [E, u] = GLnlse(matL, V, sigma, dt, epsilon)
% Gradient flow method for NLSE
% reference: W. Bao and Q. Du. Computing the ground state solution of Bose–Einstein condensates 
%   by a normalized gradient flow. SIAM Journal on Scientific Computing, 25(5):1674–1697, 2004.
%  written by Yuwei Fan (ywfan@stanford.edu)

if nargin < 4
    dt = 1;
end
if nargin < 5
    epsilon = 1e-5;
end

N = numel(V);

phi = rand(size(V));
phi = phi / norm(phi) * sqrt(N);
phi = phi(:);

niter = 0;
maxiter = 100;
ind = find(eye(N) == 1);
while 1
    %matA = matL + eye(N)/dt + diag(V) + diag(sigma .* phi.^2);
    matA = matL;
    matA(ind) = matA(ind) + 1/dt + V(:) + sigma .* phi(:).^2;
    phi2 = matA \ (phi/dt);
    phi2 = phi2 / norm(phi2) * sqrt(N);
    niter = niter + 1;
    if norm(phi2-phi) < epsilon
        break;
    end
    if niter > maxiter
        break;
    end
    phi = phi2;
end
u = phi2;
%matA = matL + diag(V) + diag(sigma .* phi.^2);
matA = matL;
matA(ind) = matA(ind) + V(:) + sigma .* phi(:).^2;
mu_u = matA * u;
E = mu_u' * u / (u'*u);
