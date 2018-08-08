function matK = Kfun3to1d(param, i, j)
% param.x: row vector
% param.st
% param.ss
% param.Sr
% param.mut_all: row vector

h = abs(param.x(2) - param.x(1));
range = 5;

epsilon = 1e-10;
maxmu = 30; %expint(30) =3e-15

t = tic;

n_i = max(size(i));
n_j = max(size(j));

[~, st_y] = ndgrid(param.st(i), param.st(j));

ind_nz = ~((param.ss(j)<epsilon)&(param.Sr(j)<epsilon));
ind_nz = ones(n_i, 1) * ind_nz';

matK = zeros(n_i, n_j);
mut = abs(param.mut_all(i) - param.mut_all(j)');
ind = (mut>range-epsilon) & (mut<maxmu) & ind_nz; %if mutp=0, deal with them later
matK(ind) = 0.5 * expint(mut(ind)) * h;

xy = abs(param.x(i)-param.x(j)');
ind = (abs(xy) < epsilon) & ind_nz; %diagonal
matK(ind) = 2*Intgrl(h/2, st_y(ind));

ind = (xy>epsilon)&(mut < range) & ind_nz; %adjoin diagonal
matK(ind) = Intgrl2(mut(ind)-st_y(ind)*h/2, h/2, st_y(ind));
clear mutp xyp

end

function res = Intgrl(h2, mut)
res = 0.5*(h2.*mut.*expint(h2.*mut)-exp(-h2.*mut)+1)./mut;
end

function res = Intgrl2(dis, h2, mut)
res = zeros(size(dis));
dis2 = dis + 2 * h2 .* mut;
ind = (dis<30) & (dis2<30);
res(ind) = 0.5 * abs(dis2(ind).*expint(dis2(ind)) - dis(ind).*expint(dis(ind)) + exp(-dis(ind)) - exp(-dis2(ind))) ./ mut(ind);
%ind = mut > 1e-10;
%res(ind) = abs(dis2(ind).*expint(dis2(ind)) - dis(ind).*expint(dis(ind)) + exp(-dis(ind)) - exp(-dis2(ind))) ./ mut(ind);
%ind = ~ind;
%res(ind) = 2*h2.*expint(dis(ind));
end

