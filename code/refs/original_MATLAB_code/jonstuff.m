
%load homa_data.mat

% data contains Dim, x_interp, and z ... take a look:

% data contains
%   dim = 3
%   x_interp (matrix, note, this would be considered transposed)
%   

%csvwrite('x_interp.csv',x_interp)
%csvwrite('z.csv',z)

% % for i=1+ncov:length(z_hist)-ncov
% %     i
% %     break
% % end

%size(U)
%size(S)
%size(V)

%Dis(:,j) = repmat(a2, M, 1) + b2 - 2*ab;
%size(repmat(a2, M, 1))
%M
%size(b2)
%size(ab)

%size(omega)
%size(A2)

%% for looking atresults
xref1 = z(1,:)';
size(xref1)
xref1 = xref1(1:5:end,1);

% graph a portion
s = 1:4000;

hold off
plot(s,xref1(s,1),'k','linewidth',1)
hold on

plot(s,Psi(s,1)*10,'LineWidth',1.5)
plot(s,Psi(s,2)*10,'LineWidth',1.5)
plot(s,Psi(s,3)*10,'LineWidth',1.5)

%%

