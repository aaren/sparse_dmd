% 2D Channel Flow
% Written by Mihailo R. Jovanovic, September 2013
%
% Script expains how to obtain necessary data for DMD
%
% In particular, matrices of snapshots 
% X0 = [x_0 ... x_{N-1}]
% X1 = [x_1 ... x_N]
% for 2D channel flow are computed

cd channel

% Number of collocation points in y
Ny = 150;
% Reynolds number
Re = 10000;
% Imaginary unit
ii = sqrt(-1);

% Streamwise and spanwise wavenumbers kx and kz
kx = 1;
kz = 0;

% Time step
dT = 1;

% Differentiation matrices
[yvecT,DM] = chebdif(Ny+2,2);
yvec = yvecT(2:end-1);

% First derivative with homogeneous Dirichlet BCs
D1 = DM(2:Ny+1,2:Ny+1,1);
% Second derivative with homogeneous Dirichlet BCs
D2 = DM(2:Ny+1,2:Ny+1,2);

% Fourth derivative with homogeneous Dirichlet and Neumann BCs
[y,D4] = cheb4c(Ny+2);

% Identity and zero matrices 
I = eye(Ny);
Z = zeros(Ny,Ny);

% Matrix representation of Umean, Uy, Uyy
% Poiseuille flow
Umean = diag(1 - yvec.^2);
Uy = diag(-2*yvec);
Uyy = diag(-2*ones(size(yvec)));

%=====================
% Start computation %%
%=====================

% k2 := kx^2 + kz^2 
k2 = kx^2 + kz^2;
k4 = k2*k2;

% Laplacian
Delta = D2 - k2*I;
% Laplacian "squared"
Delta2 = D4 - 2*k2*D2 + k4*I;

% Orr-Sommerfeld operator
A = Delta\( (1/Re)*Delta2 + ii*kx*(Uyy - Umean*Delta) );

% Time discretization of the Orr-Sommerfeld operator
Ad = expm(A*dT); 

% E-values of the Orr-Sommerfeld operator
Eos = eig(A); 

% % Set initial condition
% x0 = zeros(Ny,1);
% 
% for n = 0:10,
%     
%     x0 = x0 + randn(1,1)*chebyshev(n,yvec).*(1 - yvec.^2).^2;
%     
% end

% Or load the initial condition used in POF paper
load x0.mat

% Number of samples
nsamp = 111;
% Number of discarded snapshots
n0 = 12;

% Matrix of snapshots
Xfull = zeros(size(x0,1),nsamp+1); % initialize size of Xfull
Xfull(:,1) = x0; % set the 1st snapshot
xtmp = x0;

% Compute the matrix of snapshots
for n = 1:nsamp,
    
    xtmp = Ad*xtmp; % advance in time
    Xfull(:,n+1) = xtmp;
    
end

% Eliminate first n0 samples
X = Xfull(:,n0:end);

% Extract matrices of consequtive snapshots
X0 = X(:,1:end-1);
X1 = X(:,2:end);

% Economy-size SVD of X0
[U,S,V] = svd(X0,'econ');

% Rank of S
r = rank(S);

% Truncated versions of U, S, and V
U = U(:,1:r);
S = S(1:r,1:r);
V = V(:,1:r);

% Determine matrix UstarX1
UstarX1 = U'*X1;

% Save necessary data for DMD 
save channel.mat UstarX1 S V dT Eos % Eos saved for visualization purposes

cd ..

