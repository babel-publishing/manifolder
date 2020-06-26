% Compute the embedding
% ***************************************************************@

%% Configuration
m = 4000;                 % starting point for sequantial processing/extension
data = z_mean.';          % set the means as the input set
M = size(data, 1);

% Choose subset of examples as reference
subidx = sort(randperm(size(z_mean,2),m));

% Choose first m examples as reference
% subidx = 1:m;

dataref = data(subidx,:);

%%
% Affinity matrix - naive computation - uncomment and use for debugging
% Dis = zeros(M, m);
% h = waitbar(0, 'Please wait');
% for i = 1:M
%     waitbar(i/M, h);
%     for j = 1:m
%          % The Mahalanobis distance
%          Dis(i,j) = [data(i,:) - dataref(j,:)] * inv_c(:,:,subidx(j)) * [data(i,:) - dataref(j,:)]';
%     end
% end
% close(h);

%%
% Affinity matrix computation
Dis = zeros(M, m);
h = waitbar(0, 'Please wait');
for j = 1:m
    waitbar(j/m, h);
    tmp1 = inv_c(:,:,subidx(j)) * dataref(j,:)';

    a2 = dataref(j,:) * tmp1;
    b2 = sum(data .* (inv_c(:,:,subidx(j)) * data')',2);
    ab = data * tmp1;
    Dis(:,j) = repmat(a2, M, 1) + b2 - 2*ab;
end
close(h);

%% Anisotropic kernel
ep = median(median(Dis)); % default scale - should be adjusted for each new realizations

A = exp(-Dis/(4*ep));
W_sml=A'*A;    
d1=sum(W_sml,1);
A1=A./repmat(sqrt(d1),M,1);
W1=A1'*A1;

d2=sum(W1,1);
A2=A1./repmat(sqrt(d2),M,1);
W2=A2'*A2;

D=diag(sqrt(1./d2));

% Compute eigenvectors
[V,E] = eigs(W2,10);
[srtdE,IE] = sort(sum(E),'descend');
Phi = D*V(:,IE(1,2:10));

% Extend reference embedding to the entire set
Psi=[];

omega=sum(A2,2);
A2_nrm=A2./repmat(omega,1,m);

for i=1:size(Phi,2)
    psi_i=A2_nrm*Phi(:,i)./sqrt((srtdE(i+1)));
    Psi=[Psi,psi_i];
end

% Since close to a degenerate case - try to rotate according to:
% A. Singer and R. R. Coifman, "Spectral ICA", ACHA 2007.
%

% % NOTE: In case there are merely 2 principal components rather than 3 
% % (according to the kernel spectrum), this correction is unnecessary 
% theta_psi = atan(0.5 * (Psi(:, 3).' * (Psi(:,1).*Psi(:,1) - Psi(:,2).*Psi(:,2)) ) / ( Psi(:,3).' * (Psi(:,1) .* Psi(:,2)))) / 2;
% Psi_1 = Psi(:, 1) * cos(theta_psi) - Psi(:,2) * sin(theta_psi);
% Psi_2 = Psi(:, 1) * sin(theta_psi) + Psi(:,2) * cos(theta_psi);
% 
% %% Plot
% 
% % Plot spectrum
% figure;
% bar(1:9, diag(E(2:10, 2:10)), 1) 
% title('The Kernel Spectrum', 'FontSize', 16);
% xlabel('i', 'FontSize', 16);
% ylabel('\lambda_i', 'FontSize', 16);
% 
% % Plot embdding
% figure; 
% subplot(2,2,1); scatter(x_ref(1,:), Psi_1); xlabel('\theta_1(t)','FontSize', 16); ylabel('\psi_1(t)','FontSize', 16); xlim([-0.5 2]); ylim([-0.05 0.05]);
% subplot(2,2,2); scatter(x_ref(1,:), Psi_2); xlabel('\theta_1(t)','FontSize', 16); ylabel('\psi_2(t)','FontSize', 16); xlim([-0.5 2]); ylim([-0.05 0.05]);
% 
% subplot(2,2,3); scatter(x_ref(2,:), Psi_1); xlabel('\theta_2(t)','FontSize', 16); ylabel('\psi_1(t)','FontSize', 16); xlim([-1 1.5]); ylim([-0.05 0.05]);
% subplot(2,2,4); scatter(x_ref(2,:), Psi_2); xlabel('\theta_2(t)','FontSize', 16); ylabel('\psi_2(t)','FontSize', 16); xlim([-1 1.5]); ylim([-0.05 0.05]);
%  