function sol = Eval1D(N, sa, ss)
h = 1 / N;
x = (1/2:N)*h;
x = x(:);

st = ss + sa;
Sr = ones(size(x));

mut_all = zeros(N,1); 
mut_all(1) = st(1)/2;
for i = 2:N
    mut_all(i) = sum(st(1:(i-1))) + st(i)/2;
end
mut_all = mut_all * h;

param.x       = x;
param.st      = st;
param.ss      = ss;
param.Sr      = Sr;
param.mut_all = mut_all;

%t = tic;
matK = Kfun3to1d(param, 1:N, 1:N);
matA = eye(N) - matK .* ss(:)';
rhs = matK * Sr(:);

sol = matA \ rhs;
%toc(t)
