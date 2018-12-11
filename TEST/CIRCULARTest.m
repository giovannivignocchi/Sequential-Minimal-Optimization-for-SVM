%% TEST INITIALIZATION
clear all;
clc;
name = 'CIRCULAR TEST';

% For reproducibility
seed = 100;
rng(seed);


saveResult = 1; % if set to 1 the results of the test will be stored
[path,fid] = initTest(saveResult, name, seed);


%% CREATE THE ARTIFICIAL DATASET
r = sqrt(rand(100,1)); % Radius
t = 2*pi*rand(100,1);  % Angle
data1 = [r.*cos(t), r.*sin(t)]; % Points
r2 = sqrt(3*rand(100,1)+1); % Radius
t2 = 2*pi*rand(100,1);      % Angle
data2 = [r2.*cos(t2), r2.*sin(t2)]; % points
xTrain = [data1;data2];
yTrain = ones(200,1);
yTrain(1:100) = -1;

% Standardize the dataset
xTrain = zscore(xTrain);

% Shuffle del dataset
s = RandStream('mt19937ar','Seed',0);
rand_pos = randperm(s, size(xTrain,1)); %array of random positions
xTrainShuffle = xTrain;
yTrainShuffle = yTrain;
for i=1:size(xTrain,1)
    yTrainShuffle(i,1) = yTrain(rand_pos(i));
    xTrainShuffle(i,:) = xTrain(rand_pos(i),:);
end

xTrain = xTrainShuffle;
yTrain = yTrainShuffle;

% build the grid over which make preiction
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(xTrain(:,1)):d:max(xTrain(:,1)),min(xTrain(:,2)):d:max(xTrain(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];



%% BUILDING MODELS

% Set parameter for the Models
C = inf;
tolerance = 10e-5; % Tolerance allowed in the violation of the KKT conditions
tau = 1e-12;
eps = 10e-5;
maxiter = 200;
kernel = 'gaussian';

if saveResult
    fprintf(fid, 'Model parameter:\n');
    fprintf(fid, 'C = %d\n', C);
    fprintf(fid, 'tolerance = %d\n', tolerance);
    fprintf(fid, 'tau = %d\n', tau);
    fprintf(fid, 'eps = %d\n', eps);
    fprintf(fid, 'maxiter = %d\n', maxiter);
    fprintf(fid, 'kernel = %s\n\n', kernel);
end

[models,figureTitle] = initModel(xTrain, yTrain, C, tolerance, eps, tau, maxiter, kernel);
output = zeros(size(xGrid,1),size(models,2));

trainingStats = cell(1, size(models,2));
predictionStats = cell(1, size(models,2));
numberOfSV = cell(1, size(models,2));

for k=1:size(models,2)
    
    tic
    models{k}.train();
    trainingStats{k} = toc;
    
    tic
    output(:,k) = models{k}.predict(xGrid);
    predictionStats{k} = toc;
    
    numberOfSV{k} = sum(models{k}.isSupportVector);    
end


% Save the current workspace
varFile = strcat(path,'\var.m');
save varFile;

%% TEST RESULTS

% Write Test statistics
for k=1:size(models,2)
    fprintf(fid, compose("------------------------------------------ %s ------------------------------------------\n", figureTitle{k}));
    
    fprintf(fid, 'Training time %f sec\n', trainingStats{k});

    fprintf(fid, 'Prediction time %f sec\n', predictionStats{k});
    
    fprintf(fid, 'Number of support vector generated: %d\n', numberOfSV{k});
end
fclose(fid);


% Plot the reults for the generated models
plotResults(saveResult, path, name, models, figureTitle, output, x1Grid, x2Grid)
