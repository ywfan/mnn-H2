Nsample = 4e4; % it may cost a very long time
N = 80;
sa = 0.2;
strength = 0.02;
min_r = 0.01;
ng = 2;

xx = (0.5:N)/N;
[x1,x2] = ndgrid(xx);
xs = [x1(:) x2(:)]'; % 2d coordinates
sr = ones(N,N);

sols = zeros(Nsample, N, N);
sses = zeros(Nsample, N, N);

t = tic;
parfor k = 1:Nsample
    if mod(k, 10) == 0
        disp(k)
        toc(t)
    end
    rr = strength * rand(ng,1) + min_r;
    u1 = rand(ng, 1);
    u2 = rand(ng, 1);
    T = 2e-3 * (rand + 1);
    while 1
        u1 = 0.2 + rand(1,ng)*0.6; %[0.2, 0.8]
        u1 = sort(u1);
        if sum(abs(diff(u1)) < 0.05) == 0
            break;
        end
    end
    while 2
        u2 = 0.2 + rand(1,ng)*0.6; %[0.2, 0.8]
        u2 = sort(u2);
        if sum(abs(diff(u2)) < 0.05) == 0
            break;
        end
    end

    musfun = @(x,y) rr(1)/(2*pi*T)*exp(-((x-u1(1)).^2+(y-u2(1)).^2)/(2*T)) + rr(2)/(2*pi*T)*exp(-((x-u1(2)).^2+(y-u2(2)).^2)/(2*T));
    mufun = @(x,y) sa + rr(1)/(2*pi*T)*exp(-((x-u1(1)).^2+(y-u2(1)).^2)/(2*T)) + rr(2)/(2*pi*T)*exp(-((x-u1(2)).^2+(y-u2(2)).^2)/(2*T));
    [sol, ss] = rte2d(N, musfun, mufun, sr);
    sol = reshape(sol, [N,N]);
    ss = reshape(ss, [N,N]);
    sols(k, :, :) = sol;
    sses(k, :, :) = ss;
end

suffix = ['rte2dv1g', int2str(ng)];
filename  = [suffix, '.h5'];
if exist(filename, 'file') == 2
    delete(filename);
end
h5create(filename, '/Input', [Nsample, N, N]);
h5write( filename, '/Input', sses); 
h5create(filename, '/Output', [Nsample, N, N]);
h5write( filename, '/Output', sols);
h5create(filename, '/sa', 1);
h5write( filename, '/sa', sa);
