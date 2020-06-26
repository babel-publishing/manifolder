% Estimate empirical densities in short windows
% ***************************************************************@

%% Configuration
H = 20;                   % nonoverlapping window length for histogram/empirical densities estimation
stepSize = 5;
nbins = 5;               % # of histogram bins

%% Concatenate 1D histograms (marginals) of each sensor in short windows 
z_hist = [];
for dim=1:N
    z_hist_dim = [];
    hist_bins = linspace(min(z(dim,:)), max(z(dim,:)), nbins);
    for i=1:floor((size(z,2)-H)/stepSize)
        interval = z(dim, 1+(i-1)*stepSize: (i-1)*stepSize+H);
        z_hist_dim(:, end+1) = (hist(interval, hist_bins))';
    end
    z_hist = [z_hist; z_hist_dim];
end

%% A reference signal for comparison (a sample per window)

%jd - this is not needed now?
%x_ref = [];
%x_ref(1,:) = downsample(x_interp(1,1+H/2:end-H/2), stepSize,floor(stepSize/2));
%x_ref(2,:) = downsample(x_interp(2,1+H/2:end-H/2), stepSize,floor(stepSize/2));
