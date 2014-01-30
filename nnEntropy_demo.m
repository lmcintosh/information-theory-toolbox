% nnEntropy.m


d    = 1:20;
n    = 1000;
true = 0.5*log2((2*pi*exp(1)).^d);
est  = zeros(20,1);

for k = 1:20
    samples = randn(n,k);
    
    Ak = (k*pi^(k/2))/gamma(k/2+1);
    distmat = squareform(pdist(samples,'euclidean'));
    distmat = distmat + 1e2*eye(n);
    md      = min(distmat);
    
    est(k) = k*mean(log2(md)) + log2(n*Ak/k) - psi(1)/log(2);
    
end


figure;
plot(true,'g--'), hold on
plot(est,'ro')
    
    