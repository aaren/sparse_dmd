% Script illustrates how to reproduce figures  
% shown in our paper and on this website

% |xdmd| vs frequency
plot(imag(log(Edmd)),abs(xdmd),'ko')
xlab = xlabel('frequency','interpreter','tex')
set(xlab,'FontName','cmmr10','FontSize',26)
ylab = ylabel('amplitude','interpreter','tex')
set(ylab,'FontName','cmmr10','FontSize',26)
h = get(gcf,'CurrentAxes'); 
set(h,'FontName','cmr10','FontSize',20,'xscale','lin','yscale','lin')

% |xdmd| vs real part
plot(real(log(Edmd)),abs(xdmd),'ko')
xlab = xlabel('real','interpreter','tex')
set(xlab,'FontName','cmmr10','FontSize',26)
ylab = ylabel('amplitude','interpreter','tex')
set(ylab,'FontName','cmmr10','FontSize',26)
h = get(gcf,'CurrentAxes'); 
set(h,'FontName','cmr10','FontSize',20,'xscale','lin','yscale','lin')

% Performance loss for the polished vector of amplitudes vs gamma 
semilogx(answer.gamma,answer.Ploss,'ko','LineWidth',1,'MarkerSize',7)
xlab = xlabel('\gamma','interpreter','tex')
set(xlab,'FontName','cmr10','FontSize',26)
ylab = ylabel('performance loss (%)','interpreter','tex')
set(ylab,'FontName','cmr10','FontSize',26)
h = get(gcf,'CurrentAxes'); 
set(h,'FontName','cmr10','FontSize',20)
axis([answer.gamma(1) answer.gamma(end) 0 1.05*answer.Ploss(end)])

% Number of non-zero amplitudes vs gamma
semilogx(answer.gamma,answer.Nz,'ko','LineWidth',1,'MarkerSize',7)
xlab = xlabel('\gamma','interpreter','tex')
set(xlab,'FontName','cmr10','FontSize',26)
ylab = ylabel('N_z','interpreter','tex')
set(ylab,'FontName','cmr10','FontSize',26)
h = get(gcf,'CurrentAxes'); 
set(h,'FontName','cmr10','FontSize',20)
axis([answer.gamma(1) answer.gamma(end) 0 1.05*answer.Nz(1)])

% Spectrum of DT system for a certain value of gamma
m = 50; 
answer.Nz(m) % number of non-zero amplitudes
ival = find(answer.xsp(:,m));
plot(real(Edmd),imag(Edmd),'ko',real(Edmd(ival)),imag(Edmd(ival)),'r+', ...
    'LineWidth',1,'MarkerSize',7)
xlab = xlabel('Re(\mu_i)','interpreter','tex')
set(xlab,'FontName','cmr10','FontSize',26)
ylab = ylabel('Im(\mu_i)','interpreter','tex')
set(ylab,'FontName','cmr10','FontSize',26)
h = get(gcf,'CurrentAxes'); 
set(h,'FontName','cmr10','FontSize',20,'xscale','lin','yscale','lin')
hold
% plot a unit circle
format compact                    % tighten loose format
format long e                     % make numerical output in double precision
theta = linspace(0,2*pi,100);     % create vector theta
x = cos(theta);                   % generate x-coordinate
y = sin(theta);                   % generate y-coordinate
plot(x,y,'b--','LineWidth',1);    % plot unit circle
axis('equal');
hold

% CT spectrum of channel flow problem for a certain value of gamma
m = 50; 
answer.Nz(m) % number of non-zero amplitudes
ival = find(answer.xsp(:,m));
plot(real(Eos),imag(Eos),'b*', ...
    real(Ect),imag(Ect),'ko', ...
    real(Ect(ival)),imag(Ect(ival)),'r+', ...
    'LineWidth',1.,'MarkerSize',7)
xlab = xlabel('Re(\mu_i)','interpreter','tex')
set(xlab,'FontName','cmr10','FontSize',26)
ylab = ylabel('Im(\mu_i)','interpreter','tex')
set(ylab,'FontName','cmr10','FontSize',26)
h = get(gcf,'CurrentAxes'); 
set(h,'FontName','cmr10','FontSize',20,'xscale','lin','yscale','lin')
axis([-0.8 0.1 -1.1 -0.1])

% |xdmd| and |xpol| vs frequency for a certain value of gamma
% amplitudes in log scale 
m = 50; 
answer.Nz(m) % number of non-zero amplitudes
ival = find(answer.xsp(:,m));
semilogy(imag(log(Edmd)),abs(xdmd),'ko', ...
     imag(log(Edmd(ival))),abs(answer.xpol(ival,m)),'r+', ...
    'LineWidth',1,'MarkerSize',7)
xlab = xlabel('frequency','interpreter','tex')
set(xlab,'FontName','cmr10','FontSize',26)
ylab = ylabel('amplitude','interpreter','tex')
set(ylab,'FontName','cmr10','FontSize',26)
h = get(gcf,'CurrentAxes'); 
set(h,'FontName','cmr10','FontSize',20)

% Determine data for performance loss vs number of dmd modes plots
Nz(1) = answer.Nz(1);
Ploss(1) = answer.Ploss(1);
ind = 1;

for i = 1:length(answer.gamma)-1,
    
    if (answer.Nz(i) == answer.Nz(i+1)),
        
        ind = ind;
        
    else 
        
        ind = ind+1;
        Nz(ind) = answer.Nz(i+1);
        Ploss(ind) = answer.Ploss(i+1);
        
    end
   
    
end

clear ind

% Performance loss vs number of dmd modes
plot(Nz,Ploss,'ko')
xlab = xlabel('number of dmd modes','interpreter','tex')
set(xlab,'FontName','cmr10','FontSize',26)
ylab = ylabel('performance loss (%)','interpreter','tex')
set(ylab,'FontName','cmr10','FontSize',26)
h = get(gcf,'CurrentAxes'); 
set(h,'FontName','cmr10','FontSize',20)
axis([Nz(end) Nz(1) 0 1.05*Ploss(end)])
