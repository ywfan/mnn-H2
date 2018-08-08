N = 320;
Nsample = 4e4;
ng = 2;
sa = 0.2;
strength = 0.2;
min_r = 0.1;

x = (1/2:N) / N;
x = x(:);

sol = zeros(Nsample, N);
ss  = zeros(Nsample, N);
rhos = zeros(Nsample, ng);
us = zeros(Nsample, ng);
Ts = zeros(Nsample, 1);

t = tic;
parfor k = 1:Nsample
    if mod(k, 1e3) == 0
        disp(k)
        toc(t)
    end
    rr = strength * rand(1,ng) + min_r; 
    uu = zeros(1,ng);
    while 1
        uu = 0.2 + rand(1,ng)*0.6; %[0.2, 0.8]
        uu = sort(uu);
        if sum(abs(diff(uu)) < 0.05) == 0
            break;
        end
    end

    T = 2e-3 * (1+rand);
    rhos(k,:) = rr;
    us(k,:) = uu;
    Ts(k) = T;

    s = zeros(size(x));
    for g = 1:ng
        s = s + gaussian(rr(g), uu(g), T, x);
    end

    sol(k,:) = Eval1D(N, sa, s);
    ss(k,:) = s;
end

suffix = ['rte1dv2g', int2str(ng)];
filename  = [suffix, '.h5'];
if exist(filename, 'file') == 2
    delete(filename);
end
h5create(filename, '/Input', [N, Nsample]);
h5write( filename, '/Input', ss'); 
h5create(filename, '/Output', [N, Nsample]);
h5write( filename, '/Output', sol');
h5create(filename, '/rho', [ng, Nsample]);
h5write( filename, '/rho', rhos');
h5create(filename, '/u', [ng, Nsample]);
h5write( filename, '/u', us');
h5create(filename, '/T', [ Nsample]);
h5write( filename, '/T', Ts);
h5create(filename, '/sa', 1);
h5write( filename, '/sa', sa);

function res = gaussian(r, u, T, xx)
res = r / sqrt(2*pi*T) * exp( -(xx-u).^2 / (2*T));
end
