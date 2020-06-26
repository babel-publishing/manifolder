% Cluster embedding and generate figures and output files
% ***************************************************************@
%% Configuration
numClusters = 7;
intrinsicDim = Dim;  % can be varied slightly but shouldn't be much larger than Dim

%% Clusters
IDX = kmeans(Psi(:,1:intrinsicDim),numClusters);

%% Figures
figure
subplot(2,2,1);
scatter3(Psi(:,1),Psi(:,2),Psi(:,3),20,1:size(Psi,1))
title('Color by Time');
axis image

subplot(2,2,2);
scatter3(Psi(:,1),Psi(:,2),Psi(:,3),20,IDX)
title('Color by Cluster');
axis image

if size(x_ref,1)>=1
subplot(2,2,3);
scatter3(Psi(:,1),Psi(:,2),Psi(:,3),20,x_ref(1,:))
title('Color by x_ref(1,:)');
axis image
end

if size(x_ref,1)>=2
subplot(2,2,4);
scatter3(Psi(:,1),Psi(:,2),Psi(:,3),20,x_ref(2,:))
title('Color by x_ref(2,:)');
axis image
end

%% Output Files
time = (1:size(z,2))';
time = time(1+H/2:stepSize:end-H/2);

data = z';
data = downsample(data(1+H/2:end-H/2,:), stepSize,floor(stepSize/2));

disp('Currently assuming x_ref(2,:) contains old labels');
cHeader = {'Time' 'Data 1' 'Data 2' 'Data 3' 'Data 4' 'Data 5' 'Data 6' 'Data 7' 'Data 8' 'Old Labels' 'New Labels'}; %dummy header
commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader); %cHeader in text with commas

fid = fopen('SolarWindTimeSeriesClustering.csv','w');
fprintf(fid,'%s\n',textHeader);
fclose(fid);
dlmwrite('SolarWindTimeSeriesClustering.csv',[time data x_ref(2,:)' IDX],'-append');

save('solar_wind_data_embedding_and_parameters.mat','H','stepSize','nbins','ncov','m','subidx','Psi');