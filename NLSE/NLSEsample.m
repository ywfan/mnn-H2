Nx = 320;
Nsample = 4e4;
ng = 2; % number of gaussian

h = 1 / Nx;
x = (1:Nx)/Nx;

matL = 2 * eye(Nx) - diag(linspace(1,1,Nx-1),1) - diag(linspace(1,1,Nx-1),-1);
matL(1,end) = -1;
matL(end,1) = -1;
matL = sparse(matL);
matL = matL / h^2;

coea = zeros(Nsample, Nx);
sols = zeros(Nsample, Nx);
Es = zeros(Nsample, 1);

sigma = 10;

t = tic;
ns = 1;
ca = zeros(1,Nx);

min_r = 10;
strength = 30;
parfor ns = 1:Nsample
    if(mod(ns, 1000) == 0)
        disp(ns);
        toc(t)
    end

    rr = strength * rand(ng,1) + min_r;
    uu = zeros(ng,1);
    while 1
        uu = rand(ng, 1);
        uu = sort(uu);
        uu = [uu;uu(1)+1]; 
        if sum(abs(diff(uu)) < 0.05) == 0
            uu = uu(1:ng);
            break;
        end
    end
    
    T = 2e-3 * (rand + 1);

    ca = zeros(size(x));
    for g = 1:ng
        ca = ca -gaussian(rr(g), uu(g), T, x) - gaussian(rr(g), uu(g)+1, T, x) - gaussian(rr(g), uu(g)-1, T, x);
    end
    ca = ca - min(ca) + 1;

    [E, u] = GLnlse(matL, ca, sigma);
    coea(ns, :) = ca;
    Es(ns) = E;
    sols(ns, :) = u;
    if sum(u) < 0
        sols(ns, :) = -u;
    end
end

suffix = ['nlse2v', int2str(ng)];
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
h5create(fileinput,  '/Input', [Nx, Nsample]);
h5write( fileinput,  '/Input', coea'); 
h5create(fileoutput, '/Output', [Nx, Nsample]);
h5write( fileoutput, '/Output', sols');
h5create(fileoutput, '/E', [1, Nsample]);
h5write( fileoutput, '/E', Es');

function res = gaussian(r, u, T, xx)
res = r / sqrt(2*pi*T) * exp( -(xx-u).^2 / (2*T));
end
