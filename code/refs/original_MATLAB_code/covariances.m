% Estimate local covariance matrices
% ***************************************************************@

%% Configuration
% ncov = 10;                % size of neighborhood for covariance 
ncov = 40;

%% Covariance estimation
z_mean = zeros(size(z_hist)); % Store the mean histogram in each local neighborhood
inv_c = zeros(N*length(hist_bins), N*length(hist_bins), length(z_hist)); % Store the inverse covariance matrix of histograms in each local neighborhood
for i=1+ncov:length(z_hist)-ncov
    % Estimate covariance in short time windows
    win = z_hist(:, i-ncov:i+ncov-1);
    c = cov(win');

    % Denoise via projection on "known" # of dimensions
    [U S V] = svd(c);
    inv_c(:,:,i) = U(:,1:Dim) * inv(S(1:Dim,1:Dim)) * V(:,1:Dim)';
    z_mean(:, i) = mean(win,2);
end
