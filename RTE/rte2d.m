function [sol, mus] = rte2d(n, musfun, mufun, sr)
    xx = (0.5:n)/n;
    h = 1 / n;
    [x1,x2] = ndgrid(xx);
    xs = [x1(:) x2(:)]'; % 2d coordinates
    N = n^2;
    mus = musfun(xs(1,:), xs(2,:))';

    intgrl = zeros(N,1);
    [lvar2, weight2] = GaussLegendre_2(15);
    lvar2 = (1+lvar2) * h/4; weight2 = weight2 * h/4; %[-1,1] -> [0, h/2]
    [lvar2x, lvar2y] = ndgrid(lvar2);
    lvar2x = lvar2x(:); lvar2y = lvar2y(:);
    weight2 = kron(weight2, weight2');
    weight2 = weight2(:);
    length_d = sqrt(lvar2x.^2+lvar2y.^2);
    intgrl = sum(bsxfun(@times, (exp(-bsxfun(@times, ...
        GaussLegendreQuadrature(@(l) mufun(bsxfun(@minus, xs(1,:), l*lvar2x), bsxfun(@minus, xs(2,:), l*lvar2y))), ...
        length_d)) - 1), weight2./length_d), 1)';
    intgrl = intgrl + h*log(1+sqrt(2));
    intgrl = 2/pi * intgrl;

    matK = Kfun(xs(:,:),xs(:,:), mufun)/N; % K*h^2

    nn = 1:N;
    [indx, indy] = ndgrid(nn);
    matK(indx==indy) = intgrl;
    rhs = matK * sr(:);

    matA = eye(N,N) - matK .* mus(:);

    sol = matA \ rhs;
end

function K = Kfun(x,y, mufun)
    dx = bsxfun(@minus,x(1,:)',y(1,:));
    dy = bsxfun(@minus,x(2,:)',y(2,:));
    len1 = size(x,2);
    len2 = size(y,2);
    xinitial = repmat(x(1,:)',1,len2);
    yinitial = repmat(x(2,:)',1,len2);

    K = GaussLegendreQuadrature(@(l) mufun(xinitial-l.*dx, yinitial-l.*dy));

    %intQ2 = integral(@(l) mufun(xinitial-l.*dx, yinitial-l.*dy), 0, 1, 'ArrayValued', true);
    %disp(norm(K-intQ2)/norm(intQ2));
    %     ERROR IS E-16

    K = sqrt(dx.^2+dy.^2).*K;
    K = 1/(2*pi)*rdivide(exp(-K),sqrt( dx.^2+dy.^2 ));
end

%%%Gauss-Legendre quadrature: x = 0..1
function fval = GaussLegendreQuadrature(fun)
    Q = 5;  %number of quadrature points in Gauss-Legendre quadrature
    [lvar, weight] = GaussLegendre_2(Q);
    lvar = (1 + lvar) / 2;
    weight = weight / 2;
    fval = 0;
    for i = 1:Q
        fval = fval + weight(i) .* fun(lvar(i));
    end
end
