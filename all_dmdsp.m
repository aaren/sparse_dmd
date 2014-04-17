%% Function that explains how to run dmdsp
%
% Written by Mihailo R. Jovanovic, September 2013
%
% Notation: 
%
% Matrices of snapshots 
% X0 = [x_0 ... x_{N-1}]
% X1 = [x_1 ... x_N]
%
% Economy-size SVD of X0 
% X0 = U*S*V'
%
% Inputs: matrices U'*X1, S, and V (for a specified flow type)
% 
% Outputs:
%
% Fdmd - optimal matrix on the subspace spanned by the POD modes U of X0
% Edmd - eigenvalues of Fdmd
% Ydmd - eigenvectors of Fdmd
% xdmd - optimal vector of DMD amplitudes
% answer - gamma-parameterized structure containing output of dmdsp

gamma_grd = 200

% Load data
% Matrices U'*X1, S, and V
% Sampling period dT
% E-values of the Orr-Sommerfeld operator: Eos
load channel/channel.mat
% Sparsity-promoting parameter gamma
% Lower and upper bounds relevant for this flow type
gammaval = logspace(log10(0.15),log10(160),gamma_grd);

% Matrix Vstar 
Vstar = V';

% The number of snapshots
N = size(Vstar,2);

% Optimal DMD matrix resulting from Schmid's 2010 algorithm
% Fdmd = U'*X1*V*inv(S)
Fdmd = (UstarX1*V)/S;
% Determine the rank of Fdmd
r = rank(Fdmd); % set the number of modes
% E-value decomposition of Fdmd 
[Ydmd,Ddmd] = eig(Fdmd); 
Edmd = diag(Ddmd); % e-values of the discrete-time system

% Form Vandermonde matrix
Vand = zeros(r,N);
zdmd = Edmd;

for i = 1:N,
    
    Vand(:,i) = zdmd.^(i-1);
    
end

% Determine optimal vector of amplitudes xdmd 
% Objective: minimize the least-squares deviation between 
% the matrix of snapshots X0 and the linear combination of the dmd modes
% Can be formulated as:
% minimize || G - L*diag(xdmd)*R ||_F^2
L = Ydmd;
R = Vand;
G = S*Vstar;

% Form matrix P, vector q, and scalar s
% J = x'*P*x - q'*x - x'*q + s
% x - optimization variable (i.e., the unknown vector of amplitudes)
P = (L'*L).*conj(R*R');
q = conj(diag(R*G'*L));
s = trace(G'*G);

% Cholesky factorization of P
Pl = chol(P,'lower');

% Optimal vector of amplitudes xdmd
xdmd = (Pl')\(Pl\q);

% Sparsity-promoting algorithm starts here

% Set options for dmdsp
options = struct('rho',1,'maxiter',10000,'eps_abs',1.e-6,'eps_rel',1.e-4);

% Call dmdsp
%
% Inputs:  (1) matrix P
%              vector q 
%              scalar s
%              sparsity promoting parameter gamma
%          
%          (2) options 
%
%              options.rho     - augmented Lagrangian parameter rho
%              options.maxiter - maximum number of ADMM iterations
%              options.eps_abs - absolute tolerance 
%              options.eps_rel - relative tolerance 
%
% Output:  answer - gamma-parameterized structure containing
%
%          answer.gamma - sparsity-promoting parameter gamma
%          answer.xsp   - vector of amplitudes resulting from (SP)
%          answer.xpol  - vector of amplitudes resulting from (POL)
%          answer.Jsp   - J resulting from (SP)
%          answer.Jpol  - J resulting from (POL)
%          answer.Nz    - number of nonzero elements of x 
%          answer.Ploss - optimal performance loss 100*sqrt(J(xpol)/J(0))

%% Sparsity-Promoting Dynamic Mode Decomposition (DMDSP)
%
% Written by Mihailo R. Jovanovic, August 2012
%
% The alternating direction method of multipliers (ADMM)
% is used to solve the sparsity-promoting DMD problem:
%
% minimize  J(x) + gamma * || x ||_1              (SP)
%
% J(x) = || G - L diag{x} R ||_F^2
%
% complex-valued data G, L, R; optimization variable x
%
% J(x) can be equivalently written as
% J(x) = x'*P*x - 2*real(q'*x) + s 
% 
% P = (L'*L).*conjugate(R*R')
% q = conjugate(diag(R*G'*L))
% s = trace(G'*G)
%
% For the identified sparsity pattern, the optimal vector of 
% amplitudes is found as the answer to the following 
% structured quadratic optimization problem:
% 
%       minimize    J(x)                          
%                                                  (POL) 
%       subject to  E'*x = 0 
%
% where the columns of the matrix E are the unit vectors 
% whose non-zero elements correspond to zero components of x
%
% Syntax:
%
% answer = dmdsp(P,q,s,gammaval,options)
%
% Inputs:  (1) matrix P
%              vector q 
%              scalar s
%              sparsity promoting parameter gamma
%          
%          (2) options 
%
%              options.rho     - augmented Lagrangian parameter rho
%              options.maxiter - maximum number of ADMM iterations
%              options.eps_abs - absolute tolerance 
%              options.eps_rel - relative tolerance 
%
%              If options argument is omitted, the default values are set to
%
%              options.rho = 1
%              options.maxiter = 10000
%              options.eps_abs = 1.e-6
%              options.eps_rel = 1.e-4
%
% Output:  answer - gamma-parameterized structure containing
%
%          answer.gamma - sparsity-promoting parameter gamma
%          answer.xsp   - vector of amplitudes resulting from (SP)
%          answer.xpol  - vector of amplitudes resulting from (POL)
%          answer.Jsp   - J resulting from (SP)
%          answer.Jpol  - J resulting from (POL)
%          answer.Nz    - number of nonzero elements of x 
%          answer.Ploss - optimal performance loss 100*sqrt(J(xpol)/J(0))
%
% Additional information:
%
% http://www.umn.edu/~mihailo/software/dmdsp/

% Data preprocessing
rho = options.rho;
Max_ADMM_Iter = options.maxiter;
eps_abs = options.eps_abs;
eps_rel = options.eps_rel;

% Number of optimization variables
n = length(q);
% Identity matrix
I = eye(n);

% Allocate memory for gamma-dependent output variables
answer.gamma = gammaval;
answer.Nz    = zeros(1,length(gammaval)); % number of non-zero amplitudes 
answer.Jsp   = zeros(1,length(gammaval)); % square of Frobenius norm (before polishing)
answer.Jpol  = zeros(1,length(gammaval)); % square of Frobenius norm (after polishing)
answer.Ploss = zeros(1,length(gammaval)); % optimal performance loss (after polishing)
answer.xsp   = zeros(n,length(gammaval)); % vector of amplitudes (before polishing)
answer.xpol  = zeros(n,length(gammaval)); % vector of amplitudes (after polishing)

% Cholesky factorization of matrix P + (rho/2)*I
Prho = (P + (rho/2)*I);
Plow = chol(Prho,'lower');
Plow_star = Plow';
pause

for i = 1:length(gammaval),
    
    gamma = gammaval(i);
    
    % Initial conditions
    y = zeros(n,1); % Lagrange multiplier
    z = zeros(n,1); % copy of x
    
    % Use ADMM to solve the gamma-parameterized problem  
    for ADMMstep = 1 : Max_ADMM_Iter,
          
        % x-minimization step
        u = z - (1/rho)*y;
        % x = (P + (rho/2)*I)\(q + (rho)*u)
        xnew = Plow_star\(Plow\(q + (rho/2)*u));
        
        % z-minimization step       
        a = (gamma/rho)*ones(n,1);
        v = xnew + (1/rho)*y;
        % Soft-thresholding of v
        znew = ( (1 - a ./ abs(v)) .* v ) .* (abs(v) > a);
        
        % Primal and dual residuals
        res_prim = norm(xnew - znew);
        res_dual = rho * norm(znew - z);
        
        % Lagrange multiplier update step
        y = y + rho*(xnew - znew);
        
        % Stopping criteria
        eps_prim = sqrt(n) * eps_abs + eps_rel * max([norm(xnew),norm(znew)]);
        eps_dual = sqrt(n) * eps_abs + eps_rel * norm(y);
        
        if (res_prim < eps_prim) && (res_dual < eps_dual)
            break;
        else
            z = znew;
        end

    end             
    
    % Record output data
    answer.xsp(:,i) = z; % vector of amplitudes
    answer.Nz(i) = nnz(answer.xsp(:,i)); % number of non-zero amplitudes
    answer.Jsp(i) = real(z'*P*z) - 2*real(q'*z) + s; % Frobenius norm (before polishing)
    
    % Polishing of the nonzero amplitudes 
    % Form the constraint matrix E for E^T x = 0
    ind_zero = find( abs(z) < 1.e-12); % find indices of zero elements of z
    m = length(ind_zero); % number of zero elements
    E = I(:,ind_zero);
    E = sparse(E);
    E
    
    % Form KKT system for the optimality conditions 
    KKT = [P, E; E', zeros(m,m)];
    rhs = [q; zeros(m,1)];
    
    % Solve KKT system
    sol = KKT \ rhs;
    
    % Vector of polished (optimal) amplitudes
    xpol = sol(1:n);
    
    % Record output data
    answer.xpol(:,i) = xpol; 
    % Polished (optimal) least-squares residual
    answer.Jpol(i) = real(xpol'*P*xpol) - 2*real(q'*xpol) + s; 
    % Polished (optimal) performance loss 
    answer.Ploss(i) = 100*sqrt(answer.Jpol(i)/s);
    
    i
    
end     
