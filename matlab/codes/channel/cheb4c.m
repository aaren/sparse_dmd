function [x, D4] = cheb4c(N)

%  The function [x, D4] =  cheb4c(N) computes the fourth 
%  derivative matrix on Chebyshev interior points, incorporating 
%  the clamped boundary conditions u(1)=u'(1)=u(-1)=u'(-1)=0.
%
%  Input:
%  N:     N-2 = Order of differentiation matrix.  
%               (The interpolant has degree N+1.)
%
%  Output:
%  x:      Interior Chebyshev points (vector of length N-2)
%  D4:     Fourth derivative matrix  (size (N-2)x(N-2))
%
%  The code implements two strategies for enhanced 
%  accuracy suggested by W. Don and S. Solomonoff in 
%  SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
%  The two strategies are (a) the use of trigonometric 
%  identities to avoid the computation of differences 
%  x(k)-x(j) and (b) the use of the "flipping trick"
%  which is necessary since sin t can be computed to high
%  relative precision when t is small whereas sin (pi-t) cannot.
   
%  J.A.C. Weideman, S.C. Reddy 1998.

    I = eye(N-2);                   % Identity matrix.
    L = logical(I);                 % Logical identity matrix.

   n1 = floor(N/2-1);               % n1, n2 are indices used 
   n2 = ceil(N/2-1);                % for the flipping trick.

    k = [1:N-2]';                   % Compute theta vector.
   th = k*pi/(N-1);                 

    x = sin(pi*[N-3:-2:3-N]'/(2*(N-1))); % Compute interior Chebyshev points.

    s = [sin(th(1:n1)); flipud(sin(th(1:n2)))];   % s = sin(theta)
                               
alpha = s.^4;                       % Compute weight function
beta1 = -4*s.^2.*x./alpha;          % and its derivatives.
beta2 =  4*(3*x.^2-1)./alpha;   
beta3 = 24*x./alpha;
beta4 = 24./alpha;
    B = [beta1'; beta2'; beta3'; beta4'];

    T = repmat(th/2,1,N-2);                
   DX = 2*sin(T'+T).*sin(T'-T);     % Trigonometric identity 
   DX = [DX(1:n1,:); -flipud(fliplr(DX(1:n2,:)))];   % Flipping trick. 
DX(L) = ones(N-2,1);                % Put 1's on the main diagonal of DX.

   ss = s.^2.*(-1).^k;              % Compute the matrix with entries
    S = ss(:,ones(1,N-2));          % c(k)/c(j)
    C = S./S';                      

    Z = 1./DX;                      % Z contains entries 1/(x(k)-x(j)).
 Z(L) = zeros(size(x));             % with zeros on the diagonal.

    X = Z';                         % X is same as Z', but with 
 X(L) = [];                         % diagonal entries removed.
    X = reshape(X,N-3,N-2);

    Y = ones(N-3,N-2);              % Initialize Y and D vectors.
    D = eye(N-2);                   % Y contains matrix of cumulative sums,
                                    % D scaled differentiation matrices.
for ell = 1:4
          Y = cumsum([B(ell,:); ell*Y(1:N-3,:).*X]); % Recursion for diagonals
          D = ell*Z.*(C.*repmat(diag(D),1,N-2)-D);   % Off-diagonal
       D(L) = Y(N-2,:);                              % Correct the diagonal
DM(:,:,ell) = D;                                     % Store D in DM
end

   D4 = DM(:,:,4);                  % Extract fourth derivative matrix
