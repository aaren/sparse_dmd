%% chebyshev.m 

function R = chebyshev(n,y)
%% Compute n-th Chebyshev polynomial 

%% n--index that tells what polynomial we want to compute
%% y--grid

[row,col] = size(y);
T0 = ones(row,1);
T1 = y;

if (n == 0)
    R = T0;
  elseif(n == 1)
    R = T1;
  else 
    for i = 1:n-1
      T2 = 2*y.*T1 - T0;
      T0 = T1;
      T1 = T2;
    end
    R = T2;
end
