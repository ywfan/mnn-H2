Nx = 80;
Nsample = 4e4;
ng = 2; % number of gaussian

h = 1 / Nx;
x = (1:Nx)/Nx;
[x1, x2] = ndgrid(x);

matL = zeros(Nx,Nx,Nx,Nx);
for i = 1:Nx
    for j = 1:Nx
        matL(i,j, i,j) = 4;
        matL(i,j, mymod(i+1,Nx), j) = -1;
        matL(i,j, mymod(i-1,Nx), j) = -1;
        matL(i,j, i, mymod(j+1,Nx)) = -1;
        matL(i,j, i, mymod(j-1,Nx)) = -1;
    end
end

matL= reshape(matL, [Nx^2, Nx^2]);
matL = sparse(matL);
matL = matL/h^2;

coea = zeros(Nsample, Nx, Nx);
sols = zeros(Nsample, Nx, Nx);
Es = zeros(Nsample, 1);

sigma = 10;

t = tic;
ns = 1;

min_r = 1;
strength = 5;
for ns = 1:Nsample
    if(mod(ns, 100) == 0)
        disp(ns);
        toc(t)
    end

    rr = strength * rand(ng,1) + min_r;
    u1 = rand(ng, 1);
    u2 = rand(ng, 1);
    T = 2e-3 * (rand + 1);
    while 1
        u1 = rand(ng, 1);
        u1 = sort(u1);
        u1 = [u1;u1(1)+1]; 
        if sum(abs(diff(u1)) < 0.05) == 0
            u1 = u1(1:ng);
            break;
        end
    end
    while 2
        u2 = rand(ng, 1);
        u2 = sort(u2);
        u2 = [u2;u2(1)+1]; 
        if sum(abs(diff(u2)) < 0.05) == 0
            u2 = u2(1:ng);
            break;
        end
    end

    ca = zeros(size(x1));
    for g = 1:ng
        ca = ca -gaussian(rr(g), u1(g), u2(g), T, x1, x2) ...
         -gaussian(rr(g), u1(g)+1, u2(g), T, x1, x2) ...
         -gaussian(rr(g), u1(g)-1, u2(g), T, x1, x2) ...
         -gaussian(rr(g), u1(g), u2(g)+1, T, x1, x2) ...
         -gaussian(rr(g), u1(g)+1, u2(g)+1, T, x1, x2) ...
         -gaussian(rr(g), u1(g)-1, u2(g)+1, T, x1, x2) ...
         -gaussian(rr(g), u1(g), u2(g)-1, T, x1, x2) ...
         -gaussian(rr(g), u1(g)+1, u2(g)-1, T, x1, x2) ...
         -gaussian(rr(g), u1(g)-1, u2(g)-1, T, x1, x2);
    end
    ca = ca - min(ca(:)) + 1;

    [E, u] = GLnlse(matL, ca(:), sigma);
    u = reshape(u, [Nx, Nx]);
    coea(ns, :, :) = ca;
    Es(ns) = E;
    sols(ns, :, :) = u;
    if sum(u) < 0
        sols(ns, :, :) = -u;
    end
end

suffix = ['nlse2d', int2str(ng)];
data_path = 'data/';
if ~exist(data_path, 'dir')
    mkdir(data_path)
end
fileinput  = [data_path, 'Input_',  suffix, '.h5'];
fileoutput = [data_path, 'Output_', suffix, '.h5'];
if exist(fileinput, 'file') == 2
    delete(fileinput);
end
if exist(fileoutput, 'file') == 2
    delete(fileoutput);
end
h5create(fileinput,  '/Input', [Nsample, Nx, Nx]);
h5write( fileinput,  '/Input', coea); 
h5create(fileoutput, '/Output', [Nsample, Nx, Nx]);
h5write( fileoutput, '/Output', sols);
h5create(fileoutput, '/E', [Nsample, 1]);
h5write( fileoutput, '/E', Es);

function res = gaussian(r, u1, u2, T, x1, x2)
res = r / sqrt(2*pi*T) * exp( -((x1-u1).^2+(x2-u2).^2) / (2*T));
end

function res = mymod(i, n)
res = mod(i,n);
if res == 0
    res = n;
end
end
